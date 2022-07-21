from runset_train import parameters
from argparse import Namespace
from runset_train import train, train_for_all_ts, train_for_set_of_reg_cos

# get list of params, check validity (type and poss)
# return list of [name, [val1, val2, ...]]
def get_parameters_of_runs(params_dict):
    # First create cart prod runsets.
    vals_list = []
    cart_prod_runsets = []
    for info in params_dict.param_infos:
        cart_prod_runsets, vals_list = info.get_input_and_update_runsets(cart_prod_runsets, vals_list, params_dict)
    # Then create args list.
    args_list = []
    indRunNo = 1
    for run in cart_prod_runsets:
        kwargs = {}
        for no, name in enumerate([info.name for info in params_dict.param_infos]): kwargs[name] = run[no]
        kwargs['indRunNo'] = indRunNo # add individual run no
        kwargs['totalInds'] = len(cart_prod_runsets)
        curr_args = Namespace(**kwargs)
        args_list.append(curr_args)
        indRunNo += 1
    return args_list

def main():
    dict_file = "params_dictionary"
    params_dict = parameters.decode_arguments_dictionary(dict_file)
    args_list = get_parameters_of_runs(params_dict)
    for args in args_list:
        print(args)
        if args.tr_for_set_reg:
            train_for_set_of_reg_cos.main(args)
        elif args.tr_for_all_ts:
            train_for_all_ts.main(args)
        else:
            train.main(args=args)
if __name__ == "__main__":
    main()



