#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.zzsp import ZZSP
from parameters import parser, YML_PATH
from loss import loss_calu

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    status = "object"
    best_att, best_obj = 0, 0
    soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )
        # print("learning_rate now is: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            # if bid > 1:
            #     break
            # batch_img = batch[0].cuda()
            predict, loss = model(batch, train_pairs, status)
            # deta_soft_att = model.soft_att - soft_att_init
            # deta_soft_obj = model.soft_obj - soft_obj_init

            # print(torch.norm(deta_soft_att), torch.norm(deta_soft_obj))
            # if status in ["object", "state"]:
            #     loss += 0.1 * (2 - (torch.norm(deta_soft_att) + torch.norm(deta_soft_obj)))
            # loss = loss_calu(predict, batch, config)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()
        # scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i +1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f"{config.fusion}_epoch_{i}.pt"))

        # if i < 5:
        #     continue

        print("Evaluating val dataset:")
        logging.info("Evaluating val dataset:")
        loss_avg, val_result = evaluate(model, val_dataset, status)
        status, best_att, best_obj, best_loss, soft_att_init,  soft_obj_init = update(status, loss_avg, val_result, best_att, best_obj, best_loss, soft_att_init,  soft_obj_init, i)
        print("Loss average on val dataset: {}".format(loss_avg))
        print("Evaluating test dataset:")
        logging.info("Evaluating test dataset:")
        evaluate(model, test_dataset, status)
        if config.best_model_metric == "best_loss":
            if loss_avg.cpu().float() < best_loss:
                best_loss = loss_avg.cpu().float()
                print("Evaluating test dataset:")
                evaluate(model, test_dataset, status)
                torch.save(model.state_dict(), os.path.join(
                config.save_path, f"{config.fusion}_best.pt"
            ))
        else:
            if val_result[config.best_model_metric] > best_metric:
                best_metric = val_result[config.best_model_metric]
                print("Evaluating test dataset:")
                evaluate(model, test_dataset, status)
                torch.save(model.state_dict(), os.path.join(
                config.save_path, f"{config.fusion}_best.pt"
            ))
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(config.save_path, f"{config.fusion}_best.pt")))
            evaluate(model, test_dataset, status)

    if config.save_model:
        torch.save(model.state_dict(), os.path.join(config.save_path, f'final_model_{config.fusion}.pt'))



def update(status, loss, result, best_att, best_obj, best_loss, soft_att_init, soft_obj_init, epoch):
    # print(status, loss, result, best_att, best_obj, best_loss)
    # if status == "object":
    #     if result["obj_acc"] > best_obj:
    #         best_obj = result["obj_acc"]
    #     else:
    #         status = "state"
    #         best_att, best_obj, best_loss = 0, 0, 0
    #         soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()
    # elif status == "state":
    #     if result["attr_acc"] > best_att:
    #         best_att = result["attr_acc"]
    #     else:
    #         status = "att+obj"
    #         best_att, best_obj, best_loss = 0, 0, 0
    #         soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()
    # else:
    #     if loss > best_loss:
    #         status = "object"
    #         best_att, best_obj, best_loss = 0, 0, 0
    #         soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()
    

    if epoch // 5 % 3 == 0:
        if status != "object":
            soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()
        status = "object"
    elif epoch // 5 % 3 == 1:
        if status != "state":
            soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()
        status = "state"
    else:
        if status != "state+object":
            soft_att_init,  soft_obj_init = model.soft_att.clone(), model.soft_obj.clone()
        status = "state + object"


    print("Now status is {}".format(status))
    logging.info("Now status is {}".format(status))

    return status, best_att, best_obj, best_loss, soft_att_init,  soft_obj_init





def evaluate(model, dataset, status):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config, status)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in test_stats:
        if key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)   
    logging.info(result)
    model.train()
    return loss_avg, test_stats



def test_model(model, dataset, epoch):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_logits = torch.Tensor()
    all_pred = torch.Tensor()
    all_obj_pred = torch.Tensor()
    all_com_pred = torch.Tensor()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # if idx > 10:
            #     break
            predict, l = model(data, pairs, epoch, "test")
            logits = predict
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.type(torch.float32).cpu()
            predict = logits.topk(1, dim=1)[1].squeeze()
            all_pred = torch.cat([all_pred, predict], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )
    if epoch%3 == 0:
        obj_acc = len(all_pred[all_pred==all_obj_gt]) / len(all_pred)
        print("Test attribute accuracy: ", 0, "Test object accuracy: ", obj_acc)
    else:
        attr_acc = len(all_pred[all_pred==all_attr_gt]) / len(all_pred)
        print("Test attribute accuracy: ", attr_acc, "Test object accuracy: ", attr_acc)
    # test_stats = test.test(
    #             dataset,
    #             evaluator,
    #             all_logits,
    #             all_attr_gt,
    #             all_obj_gt,
    #             all_pair_gt,
    #             config)
    # result = ""
    # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    # for key in test_stats:
    #     if key in key_set:
    #         result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    # print(result)   
    model.train()




if __name__ == "__main__":
    config = parser.parse_args()
    set_log("logs/" + config.dataset + ".log")
    load_args(YML_PATH[config.dataset], config)
    logging.warning(config.log_id)
    logging.info(config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = ZZSP(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # if config.load_model is not False:
    #     model.load_state_dict(torch.load(config.load_model))

    os.makedirs(config.save_path, exist_ok=True)

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    print("done!")
