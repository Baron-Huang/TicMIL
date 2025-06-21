#### Department: COE, Chongqing University
#### Attempt: fitting functions for DHM_MIL models

########################## API Section #########################
import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np
from torch import optim
import pandas as pd
from Models.ViT_models.ViT import VisionTransformer
from Models.ViT_models.ViT_model_modules import ViT_Net
import random
from Models.ViT_models.ViT_model_modules import creating_ViT
from Models.Mixer_models.models.Mixer_model_modules import creating_Mixer
import warnings


warnings.filterwarnings('ignore')


########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


########################## learning functions #########################
def cnn_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-4
    elif epoch < 75:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr


def vit_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr


def vit_lr_for_breast_schedule(epoch):
    if epoch < 50:
        lr = 2e-5
    elif epoch < 75:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr


def fusion_lr_schedule(epoch):
    if epoch < 50:
        lr = 5e-5
    elif epoch < 75:
        lr = 2e-6
    else:
        lr = 1e-6
    return lr


def one_hot(org_x=None, pre_dim=3):
    one_x = np.zeros((org_x.shape[0], pre_dim))
    for i in range(org_x.shape[0]):
        one_x[i, int(org_x[i])] = 1
    return one_x



def view_results_for_clam(mil_net=None, train_loader=None, data_parallel=False, loss_fn=None, proba_mode=False,
                            gpu_device=None, proba_value=0.85, class_num = 2, bags_stat = True):
    mil_net.eval()
    train_acc = []
    train_loss = []
    pre_y_sum = np.zeros((1, class_num))
    label_sum = np.zeros(1)

    for train_img_list, train_label in train_loader:
        if data_parallel == False:
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
                for train_img in train_img_list:
                    if bags_stat == True:
                        sti_pre_y, _, _, _  = mil_net(train_img.cuda(gpu_device), train_pre_y)
                    else:
                        sti_pre_y = mil_net(train_img.cuda(gpu_device))
                    train_pre_y = torch.cat((train_pre_y, sti_pre_y))
                train_pre_y = train_pre_y[1:]
                train_loss.append(loss_fn(train_pre_y, train_label.reshape(train_label.shape[0],)).detach().cpu().numpy())
                if proba_mode == True:
                    train_pre_y = torch.softmax(train_pre_y, dim=1)
                    train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                    for proba_in in range(train_pre_label_proba.shape[0]):
                        if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < \
                                torch.tensor(proba_value).cuda(gpu_device):
                            train_pre_label_proba[proba_in] = torch.tensor(3).cuda(gpu_device)
                    train_pre_label = train_pre_label_proba
                elif proba_mode == False:
                    train_pre_label = torch.argmax(train_pre_y, dim=1)
                else:
                    print('error! Please select probability mode!!!')
                    break
        else:
            train_label = train_label.cuda()
            with torch.no_grad():
                train_pre_y = torch.zeros((1, class_num)).cuda()
                for train_img in train_img_list:
                    if bags_stat == True:
                        sti_pre_y, _ = mil_net(train_img.cuda(gpu_device))
                    else:
                        sti_pre_y = mil_net(train_img.cuda(gpu_device))
                    train_pre_y = torch.cat((train_pre_y,sti_pre_y))
                train_pre_y = train_pre_y[1:]
                train_pre_y = (train_pre_y[0:int(train_pre_y.shape[0] / 2), :]
                               + train_pre_y[int(train_pre_y.shape[0] / 2):, :]) / 2
                train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
                if proba_mode == True:
                    train_pre_y = torch.softmax(train_pre_y, dim=1)
                    train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                    for proba_in in range(train_pre_label_proba.shape[0]):
                        if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_values).cuda():
                            train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                    train_pre_label = train_pre_label_proba
                elif proba_mode == False:
                    train_pre_label = torch.argmax(train_pre_y, dim=1)
                else:
                    print('error! Please select probability mode!!!')
                    break
        pre_y_sum = np.concatenate((pre_y_sum, train_pre_y.detach().cpu().numpy()))
        label_sum = np.concatenate((label_sum, train_label.reshape(train_label.shape[0],).detach().cpu().numpy()))

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    pre_y_sum = pre_y_sum[1:, :]
    label_sum = label_sum[1:]
    return train_acc, train_loss, pre_y_sum, label_sum


def training_for_clam(mil_net=None, train_loader=None, val_loader=None, test_loader=None, proba_mode=False,
                        lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, current_lr=None, data_parallel=False,
                        weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1_.pth',
                        proba_value=0.85, class_num = 2, bags_stat = True):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_net.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        mil_net.train()
        for img_data_list, img_label in train_loader:
            if data_parallel == False:
                img_label = img_label.cuda(gpu_device)
                #PacMIL
                adv_dis_values = torch.zeros((1, 1)).cuda(gpu_device)
                #PacMIL
                pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
                for img_data in img_data_list:
                    if bags_stat == True:
                        mil_pre_y, _, _, _ = mil_net(img_data.cuda(gpu_device), pre_y)
                    else:
                        mil_pre_y = mil_net(img_data.cuda(gpu_device))
                    #print(pre_y.shape)
                    #print(mil_pre_y.shape)
                    pre_y = torch.cat((pre_y, mil_pre_y))
            else:
                img_label = img_label.cuda()
                pre_y = torch.zeros((1, class_num)).cuda()
                for img_data in img_data_list:
                    if bags_stat == True:
                        mil_pre_y, _, _ = mil_net(img_data.cuda(gpu_device), pre_y)
                    else:
                        mil_pre_y = mil_net(img_data.cuda(gpu_device))
                    pre_y = torch.cat((pre_y, mil_pre_y))
            pre_y = pre_y[1:]

            loss_value = loss_fn(pre_y, img_label.reshape(img_label.shape[0],))

            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()
        # print(ddai_net.w)
        train_acc, train_loss, _, _ = view_results_for_clam(mil_net=mil_net, train_loader=train_loader, data_parallel=data_parallel,
                                                       loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                                       proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)

        val_acc, val_loss, _, _ = view_results_for_clam(mil_net=mil_net, train_loader=val_loader, data_parallel=data_parallel,
                                                    loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                                    proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss, _, _ = view_results_for_clam(mil_net=mil_net, train_loader=test_loader, data_parallel=False,
                                                 loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                                 proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    g = mil_net.state_dict()
    torch.save(g, weight_path)

    return test_acc

def testing_for_clam(test_model=None, train_loader=None, val_loader=None, test_loader=None, gpu_device=0,
                       out_mode=None, proba_mode=False, proba_value=0.5, class_num = 2,
                       roc_save_path = None, bags_stat = True, bag_relations_path = None):
    loss_fn = nn.CrossEntropyLoss()
    train_acc, train_loss, _, _ = view_results_for_clam(mil_net=test_model, train_loader=train_loader, data_parallel=False,
                                                          loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                                          proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)


    val_acc, val_loss, _, _ = view_results_for_clam(mil_net=test_model, train_loader=val_loader, data_parallel=False,
                                                      loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)


    test_acc, test_loss, pre_y_sum, label_sum = view_results_for_clam(mil_net=test_model, train_loader=test_loader,
                                                                        data_parallel=False, loss_fn=loss_fn,
                                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                                        proba_value=proba_value, class_num=class_num,
                                                                        bags_stat=bags_stat)



    label_proba_sum = one_hot(org_x = label_sum, pre_dim = class_num)
    pre_label_sum = np.argmax(pre_y_sum, axis=1)
    #test_all_pre_proba = one_hot(org_x= test_all_pre_label, pre_dim = class_num)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)), '\n')

    print('########################## ROC and AUC results #########################')
    print(classification_report(label_sum, pre_label_sum, digits=4))

    # print('########################## auc results #########################')
    print(roc_auc_score(np.reshape(label_proba_sum, (label_proba_sum.shape[0]*label_proba_sum.shape[1])),
                    np.reshape(pre_y_sum, (pre_y_sum.shape[0]*pre_y_sum.shape[1]))))

    fpr, tpr, _ = roc_curve(np.reshape(label_proba_sum, (label_proba_sum.shape[0]*label_proba_sum.shape[1])),
                    np.reshape(pre_y_sum, (pre_y_sum.shape[0]*pre_y_sum.shape[1])))

    print(fpr.shape, tpr.shape)

    write_dict = {'fpr':fpr, 'tpr':tpr}
    roc_pd = pd.DataFrame(write_dict)
    #roc_pd.to_csv(roc_save_path)

    #relation_bags_pd = pd.DataFrame(relation_bags)
    #relation_bags_pd.to_csv(bag_relations_path)
