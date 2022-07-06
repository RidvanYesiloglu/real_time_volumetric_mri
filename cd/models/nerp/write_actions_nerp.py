import os
import torch
import numpy as np

import time

from networks import Positional_Encoder, FFN, SIREN
from utils import prepare_sub_folder, mri_fourier_transform_3d, complex2real, random_sample_uniform_mask, random_sample_gaussian_mask, save_image_3d, PSNR, check_gpu

from torchnufftexample import create_radial_mask, project_radial, backproject_radial
import sys
import torch.nn as nn
# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'device':device}
# outs_dict: {'mean_f_zk':mean_f_zk, 'log_mean_f_zk':log_mean_f_zk, 'final_f_zk_vals':final_f_zk_vals, 'final_thetas':final_thetas}
def preallruns_actions(inps_dict):
    args = inps_dict['args']
    mean_psnr = 0
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    main_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
    if torch.cuda.is_available():
        main_logs.write('GPU Total Memory [GB]: {}\n'.format(torch.cuda.get_device_properties(0).total_memory/1e9))
    else:
        main_logs.write('Using CPU.\n')
    main_logs.close()
        
    final_psnrs = np.zeros((args.noRuns))
    np.save(os.path.join(inps_dict['save_folder'],'final_psnrs'),final_psnrs)

    prev_rec = torch.from_numpy(np.load(args.prev_rec_path))
    preallruns_dict = {'mean_psnr':mean_psnr, 'final_psnrs':final_psnrs, 'prev_rec':prev_rec}
    return preallruns_dict


# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'model': model}
# outs_dict: 
def prerun_i_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    device = inps_dict['device']
    
    
    # Setup output folder
    #output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    output_subfolder = args.data
    model_name = os.path.join(inps_dict['save_folder'], output_subfolder + '/img{}_af{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
        .format(args.img_size, args.sampler['af'], \
            args.model, args.net['network_input_size'], args.net['network_width'], \
            args.net['network_depth'], args.lr_mdl, args.encoder['embedding']))
    if not(args.encoder['embedding'] == 'none'):
        model_name += '_scale{}_size{}'.format(args.encoder['scale'], args.encoder['embedding_size'])
    print(model_name)
    
    # train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_path = '/home/yesiloglu/projects/3d_tracking_nerp/output_path'
    output_directory = os.path.join(output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    #shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    #########################
    # Setup input encoder:
    encoder_tr = Positional_Encoder(args)
    # Setup model
    model_tr = SIREN(args.net)
    model_tr.cuda(args.gpu_id)
    
    model_tr.train()
    # Setup optimizer
    optim_tr = torch.optim.Adam(model_tr.parameters(), lr=args.lr_tr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    
    # print('GPU 3 aftr model load:')
    # check_gpu(3)
    ########################
    args.net['network_width']=256
    args.net['network_depth']=8
    args.net['network_output_size']=1
    # Setup input encoder:
    encoder = Positional_Encoder(args)
    # Setup model
    model = SIREN(args.net)
    model.cuda(args.gpu_id)
    
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr_mdl, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    # Setup loss function
    mse_loss_fn = torch.nn.MSELoss()
    
    # Load pretrain model
    if args.pretrain_im and ((not args.load_prev_tr) or (inps_dict['im_ind'] == 1)):
        # model_path = '../transformer_ep737_53.43dB.pt' #.format(config['data'], config['img_size'], \
        #                 #config['model'], config['net']['network_width'], config['net']['network_depth'], \
        #                 #config['encoder']['scale'])
        # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        # model_tr.load_state_dict(state_dict['net_tr'])
        # encoder_tr.B = state_dict['enc_tr'].cuda(args.gpu_id)
        # model_tr = model_tr.cuda(args.gpu_id)
        # model_tr.train()
        
        
        model_path = args.pri_im_path#.format(config['data'], config['img_size'], \
                        #config['model'], config['net']['network_width'], config['net']['network_depth'], \
                        #config['encoder']['scale'])
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        model.load_state_dict(state_dict['net'])
        encoder.B = state_dict['enc'].cuda(args.gpu_id)
        model = model.cuda(args.gpu_id)
        model.train()
        #optim.load_state_dict(state_dict['opt'])
        print('Load pretrain model: {}'.format(model_path))
        # with torch.no_grad():
        #     deformed_grid = grid + (model(train_embedding))  # [B, C, H, W, 1]
        #     deformed_prior = model_Pus(encoder_Pus.embedding(deformed_grid))
        #     plain_prior = model_Pus(encoder_Pus.embedding(grid))
        # image = torch.from_numpy(np.expand_dims(np.load('../data73/ims_tog.npy')[70],(0,-1)).astype('float32')).cuda(args.gpu_id)
        # def_psnr = - 10 * torch.log10(mse_loss_fn(deformed_prior, image)).item()
        # pla_psnr = - 10 * torch.log10(mse_loss_fn(plain_prior, image)).item()
        # print('PLain PSNR: {:.5f}, Deformed PSNR: {:.5f}'.format(pla_psnr, def_psnr))
        
        # np.save('/home/yesiloglu/projects/transformation_nerp/priors/deformed_prior.npy',deformed_prior.detach().cpu().numpy())
        # np.save('/home/yesiloglu/projects/transformation_nerp/priors/plain_prior.npy',plain_prior.detach().cpu().numpy())
        # print('Saved, exiting')
        # sys.exit()
    model_tr = nn.DataParallel(model_tr,[args.gpu_id,args.gpu2_id])
    model = nn.DataParallel(model, [args.gpu_id, args.gpu2_id])

    # Load pretrain model
    if args.load_prev_tr and (inps_dict['im_ind'] > 1):
        state_dict = torch.load(args.prev_tr_model_path, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        model.load_state_dict(state_dict['net'])
        encoder.B = state_dict['enc'].cuda(args.gpu_id)
        model = model.cuda(args.gpu_id)
        model.train()
        model_tr.load_state_dict(state_dict['net_tr'])
        encoder_tr.B = state_dict['enc_tr'].cuda(args.gpu_id)
        model_tr = model_tr.cuda(args.gpu_id)
        model_tr.train()
        print('**Load previous model: {} (from t = {})'.format(args.prev_tr_model_path, inps_dict['im_ind']))
    elif args.load_prev_tr:
        print('**Since it is time {}, prev tr was not loaded.'.format(inps_dict['im_ind']))
    
    
    def spec_loss_fn(pred_spec, gt_spec):
        '''
        spec: [B, H, W, C]
        '''
        loss = torch.mean(torch.abs(pred_spec - gt_spec))
        return loss
    
    
    # Setup data loader
    print('Load image: {}'.format(args.img_path))
    #data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])
    
    
    args.img_size = (args.img_size, args.img_size, args.img_size) if type(args.img_size) == int else tuple(args.img_size)
    slice_idx = list(range(0, args.img_size[0], int(args.img_size[0]/args.display_image_num)))
    
    #img_path = '../../data73/ims_tog.npy' if args.togOrSepNorm==1 else '../../data73/ims_sep.npy'
    print('**before image**')
    check_gpu(args.gpu_id)
    image = torch.from_numpy(np.expand_dims(np.load(args.img_path)[args.im_ind],(0,-1)).astype('float32')).cuda(args.gpu_id)
    print('**after image**')
    check_gpu(args.gpu_id)
    #print('All slices shape ', )
    #image = torch.from_numpy(np.expand_dims(np.load(args.img_path),(0,-1))).cuda(args.gpu_id)
    grid_np = np.asarray([(x,y,z) for x in range(128) for y in range(128) for z in range(64)]).reshape((1,128,128,64,3))
    grid_np = (grid_np/np.array([128,128,64])) + np.array([1/256.0,1/256.0,1/128.0])
    grid = torch.from_numpy(grid_np.astype('float32')).reshape(-1,3)
    grid = grid.cuda(args.gpu_id)
    grid.requires_grad = True
    print('Grid element size: {} and neleements: {}, size in megabytes: {}'.format(grid.element_size(), grid.nelement(), (grid.element_size()*grid.nelement())/1000000.0))

    ktraj, im_size, grid_size = create_radial_mask(args.nproj, (64,1,128,128), args.gpu_id, plot=False)
    kdata = project_radial(image, ktraj, im_size, grid_size)
    print('Ktraj shape ', ktraj.shape, 'kdata.shape ', kdata.shape)
    
    # save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))
    # save_image_3d(complex2real(train_data[1]), slice_idx, os.path.join(image_directory, "train.png"))
    # save_image_3d(complex2real(spectrum), slice_idx, os.path.join(image_directory, "spec.png"))
    print('**before emb**')
    check_gpu(args.gpu_id)
    #train_embedding_tr = encoder_tr.embedding(grid)  # [B, C, H, W, embedding*2]
    #test_embedding = encoder.embedding(grid)
    print('**before emb 2**')
    check_gpu(args.gpu_id)
    #train_embedding = encoder.embedding(grid)  # [B, C, H, W, embedding*2]
    #test_embedding_Pus = encoder_Pus.embedding(grid)
    print('**before emb3**')
    check_gpu(args.gpu_id)
    #train_embedding_Ius = encoder_Ius.embedding(grid)  # [B, C, H, W, embedding*2]
    # test_embedding_Ius = encoder_Ius.embedding(grid)
    #im_Ius = torch.from_numpy(np.load('../Ius_models/im70_10p_ep2385_27.88dB.npy').astype('float32')).cuda(args.gpu_id)
    print('**after all emb**')
    check_gpu(args.gpu_id)
    
    
    
    init_thetas_str = "Run no: {}\n".format(inps_dict['run_number']) + '\n'
    init_psnr_str = 'Initial psnr: {:.4f} \n'.format(1)
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    r_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
    r_logs.write(init_thetas_str)
    r_logs.write(init_psnr_str)
    r_logs.close()
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write(init_thetas_str)
    main_logs.write(init_psnr_str)
    main_logs.close()
    preruni_dict={'model':model, 'grid':grid, 'model_tr':model_tr,'image':image,\
                  'ktraj':ktraj, 'im_size':im_size, 'grid_size':grid_size,'image_kdata':kdata,\
                  'spec_loss_fn':spec_loss_fn,'encoder_tr':encoder_tr,\
                  'encoder':encoder, 'mse_loss_fn':mse_loss_fn, 'slice_idx':slice_idx, 'image_directory':image_directory, \
                      'checkpoint_directory':checkpoint_directory, 'optim':optim, 'optim_tr':optim_tr}

        #np.save(os.path.join(inps_dict['save_folder'], 'pretrainmodel_out'), test_output.detach().cpu().numpy())
    return preruni_dict

def print_freq_actions(inps_dict):
    args =inps_dict['args']
    # train_writer.add_scalar('train_loss', train_loss, iterations + 1)
    print("[Epoch: {}/{}] Train loss: {:.4g} ".format(inps_dict['t']+1, args.max_iter, inps_dict['losses_r'][-1]))
    
def write_freq_actions(inps_dict, preruni_dict):
    args = inps_dict['args']
    end_time = time.time()
    
    preruni_dict['model'].eval()
    with torch.no_grad():
        deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['encoder_tr'].embedding(preruni_dict['grid'])))  # [B, C, H, W, 1]
        #deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['train_embedding_tr']))  # [B, C, H, W, 1]
        output_im = preruni_dict['model'](preruni_dict['encoder'].embedding(deformed_grid))
        output_im = output_im.reshape((1,128,128,64,1))
        test_loss = preruni_dict['mse_loss_fn'](output_im, preruni_dict['image'])
        test_psnr = - 10 * torch.log10(test_loss).item()
        print('MODEL PSNR: {:.5f}'.format(test_psnr))
        test_loss = test_loss.item()

    # train_writer.add_scalar('test_loss', test_loss, iterations + 1)
    # train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
    # Must transfer to .cpu() tensor firstly for saving images
    #save_image_3d(test_output, preruni_dict['slice_idx'], os.path.join(preruni_dict['image_directory'], "recon_{}_{:.4g}dB.png".format(inps_dict['t']+1, test_psnr)))
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: {}, Time: {}, Train Loss: {:.4f}\n'.format(inps_dict['t']+1,end_time-inps_dict['start_time'],inps_dict['losses_r'][-1]))
    to_write = "[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr)
    r_logs.write(to_write)
    start_time = time.time()
    r_logs.close()
    #plt_model.plot_change_of_objective(inps_dict['f_zks_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    #plt_model.plot_change_of_loss(inps_dict['losses_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    
    np.save(os.path.join(inps_dict['save_folder'],'psnrs_{}'.format(inps_dict['run_number'])), inps_dict['psnrs_r'])

    print(to_write)
    return {'start_time':start_time}


def postrun_i_actions(inps_dict, preallruns_dict, preruni_dict):
    args = inps_dict['args']

    preruni_dict['model'].eval()
    with torch.no_grad():
        deformed_grid = preruni_dict['grid'] + (preruni_dict['model_tr'](preruni_dict['encoder_tr'].embedding(preruni_dict['grid'])))  # [B, C, H, W, 1]
        output_im = preruni_dict['model'](preruni_dict['encoder'].embedding(deformed_grid))
        output_im = output_im.reshape((1,128,128,64,1))
        test_loss = preruni_dict['mse_loss_fn'](output_im, preruni_dict['image'])
        test_psnr = - 10 * torch.log10(test_loss).item()
        print('MODEL PSNR: {:.5f}'.format(test_psnr))
        test_loss = test_loss.item()

    # train_writer.add_scalar('test_loss', test_loss, iterations + 1)
    # train_writer.add_scalar('test_psnr', test_psnr, iterations + 1)
    # Must transfer to .cpu() tensor firstly for saving images
    #save_image_3d(test_output, preruni_dict['slice_idx'], os.path.join(preruni_dict['image_directory'], "recon_{}_{:.4g}dB.png".format(inps_dict['t']+1, test_psnr)))
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: {}, Train Loss: {:.4f}\n'.format(inps_dict['t']+1,inps_dict['losses_r'][-1]))
    to_write = "[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(inps_dict['t']+1, args.max_iter, test_loss, test_psnr)
    r_logs.write(to_write)
    r_logs.close()
    #plt_model.plot_change_of_objective(inps_dict['f_zks_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    #plt_model.plot_change_of_loss(inps_dict['losses_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    
    print('**************')
    print('FINAL RES.: ' + to_write)
    preallruns_dict['mean_psnr'] += test_psnr/args.noRuns
    print('******************************************')
    
    preallruns_dict['final_psnrs'][inps_dict['run_number']] = test_psnr
    np.save(os.path.join(inps_dict['save_folder'],'final_psnrs'), preallruns_dict['final_psnrs'])
    
    np.save(os.path.join(inps_dict['save_folder'],'psnrs_{}'.format(inps_dict['run_number'])), inps_dict['psnrs_r'])
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write(to_write)
    main_logs.close()
    
    print('Encoder B[65:68,:]: ',preruni_dict['encoder'].B[65:68,:])
    

def postallruns_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    save_folder = inps_dict['save_folder']
    final_psnrs = preallruns_dict['final_psnrs']
    print("All psnr values", final_psnrs)
    main_log = 'Mean psnr over all runs: {:.6f} \n'.format(preallruns_dict['mean_psnr'])
    print(main_log)
    print('********************************TRAINING ENDED**********************************************')
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    main_logs.write(main_log)
    main_logs.close()
    
