from runset_train import train
import torch
import pickle
import os
import runset_train.parameters as parameters
import numpy as np
import sys
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

def find_total_runs(wts, sps, jcs, ts):
    curr_ind = 0
    for wt in wts:
        for sp in sps:
            for jc in jcs:
                for t in ts:
                    if wt==0 and jc != 0:
                        continue
                    curr_ind += 1

                    print(wt, sp, jc, (jc!=0))
                    
    return curr_ind
def main(args):
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    if args.end_ind == -1:
        args.end_ind = np.load(args.data_dir+args.pt+'/all_vols.npy').shape[0] - 1
        print(f'Ending index was made: {args.end_ind} (which is the last data point over time.)')
    wts = [1]
    sps = [0,1e2,1e3,1e4]
    jcs = [0,1e2,1e3,1e4]
    ts = [0,1e2]# [1e3,1e4]
    print('Experiments will be done with and without transformation.')
    print('Set of spatial regulariation coefficients:', sps)
    print('Set of Jacobian (on grid) regulariation coefficients:', jcs)
    tot_runs = find_total_runs(wts, sps, jcs, ts)
    curr_ind = 0
    for wt in wts:
        for sp in sps:
            for jc in jcs:
                for t in ts:
                    if wt==0 and jc != 0:
                        continue
                    print('**************************************************************')
                    curr_ind += 1
                    if curr_ind < 19:
                        continue
                    print('Current run number: {}/{}'.format(curr_ind, tot_runs))
                    args.conf = 'trn_wo_trns' if wt==0 else 'trn_w_trns'
                    args.use_sp_cont_reg = (sp!=0)
                    args.use_t_cont_reg = (t!=0)
                    args.use_jc_grid_reg = (jc!=0)
                    args.lambda_sp = sp
                    args.lambda_JR = jc
                    args.lambda_t = t
                    print('Configuration: {}, spatial reg. co.: {}, Jacobian reg. co.: {}, time reg co: {}'.format(args.conf, args.lambda_sp, args.lambda_JR, args.lambda_t))
                    print(f'Ending index was made: {args.end_ind} (which is the last data point over time.)')
                    for i in range(args.st_ind, args.end_ind + 1):
                        print('************************************')
                        args.im_ind = i
                        print('Train for all for loop iteration time t = {}'.format(i))
                        opts_strs = create_opts_strs([args], params_dict)
                        os.system(f'python3 -m runset_train.train{opts_strs}')
                        
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
    
    