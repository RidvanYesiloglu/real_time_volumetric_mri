# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:09:39 2022

@author: ridva
"""
from runset_train import train
import torch
import pickle
import os
import runset_train.parameters as parameters
import numpy as np

def find_prev_model_dir(args, i, params_dict):
    working_dir = '/home/yesiloglu/projects/geo_loss_nerp/cascade_models/load_prev_net'
    if i==2:
        args.lr_tr = 1e-4
    repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos], wantShort=True, params_dict=params_dict)
    save_folder = os.path.join(working_dir, 'detailed_results', 'all_'+ str(args.pt), 'vol_'+str(i-1), repr_str)
    print('save folder is ', save_folder)
    print([name for name in os.listdir(save_folder)])
    model_path = save_folder + '/' + ([name for name in os.listdir(save_folder) if name.endswith('.pt')][0])
    rec_path = save_folder + '/' + ([name for name in os.listdir(save_folder) if name.endswith('.npy')][0])
    if i==2:
        args.lr_tr = 1e-6
    return model_path, rec_path

# return a "\n" separated list
def create_opts_strs(args_list, params_dict):
    opts_strs = ""
    names_list = [info.name for info in params_dict.param_infos]
    for args in args_list:
        # First check validity of args
        parameters.check_args(args, params_dict)
        opts = ""
        for no,name in enumerate(names_list):
            if eval("args."+name) is not None:
                if params_dict.param_infos[no].typ == 'type_check.dictionary':
                    inp_dict = eval("args."+name)
                    dict_str = "{"
                    for key in inp_dict:
                        dict_str += "\"{}\":".format(key)
                        dict_str += "\"{}\",".format(inp_dict[key]) if type(inp_dict[key])==str else "{},".format(inp_dict[key])
                    dict_str = dict_str[:-1]+"}"
                    opts += " --"+name + " " + dict_str
                elif params_dict.param_infos[no].typ == 'type_check.positive_int_tuple':
                    opts += " --"+name + " " + (str(eval("args."+name))[1:-1].replace(',',''))
                elif params_dict.param_infos[no].typ == 'type_check.boolean':
                    opts = (opts + " --"+name + " 1") if eval("args."+name) else (opts + " --"+name + " 0")
                else:
                    opts += " --"+name + " " + str(eval("args."+name))
        opts_strs += opts + "\n"#":"
    return opts_strs[:-1]


def main(args=None):
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    no_of_time_pts = 1
    print('No of time pts is: {}'.format(no_of_time_pts))
    args.lr_tr = 1e-4
    for i in range(1,no_of_time_pts):
        print('Train for all for loop iteration time t = {}'.format(i))
        opts_strs = create_opts_strs([args], params_dict)
        print('opts_strts:')
        print(opts_strs)
        print('boyle')
        os.system(f'python3 -m runset_train.train from runset_train{opts_strs}')
        
    
    
    '''
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    no_of_time_pts = 30#np.load(args.img_path).shape[0]
    print('No of time pts is: {}'.format(no_of_time_pts))
    args.lr_tr = 1e-4
    for i in range(1,no_of_time_pts): #(1739,1760) (1760,1781), (1781,1802)
        print('Train for all for loop iteration time t = {}'.format(i))
        if args.load_prev_tr and (i>1):
            args.prev_tr_model_path, args.prev_rec_path = find_prev_model_dir(args,i,params_dict)
            print('Previous transformer model path is made {}.'.format(args.prev_tr_model_path))
            print('Previous rec path is made {}.'.format(args.prev_rec_path))
        #args.img_path = '../data/patient19/volume_{}.npy'.format(i)
        #train.main(args, i)
        with open('args_obj', 'wb') as f:
            pickle.dump({'args':args,'i':i}, f)
        print('Args, i, pt dumped!')
        #train_for_all.main(args)
        #exec(open("filename").read())
        os.system('python3 -m runset_train.train from runset_train')
        args.lr_tr = 1e-6'''
if __name__ == "__main__":
    main() 
    
    