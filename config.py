"""Default simulation parameters can be controlled via the command line arguments below."""
import argparse


def get_base_name(config):
    model_type = config.model_type
    target_type = config.target_type
    train_on = config.train_on
    noise_level = config.noise_level
    train_size = config.train_size
    test_size = config.test_size
    n_iters = config.n_iters
    n_epochs = config.n_epochs
    min_pass = config.min_pass
    max_pass = config.max_pass
    pass_range = (min_pass, max_pass)
    min_num = config.min_num
    max_num = config.max_num
    num_range = (min_num, max_num)
    use_loss = config.use_loss
    drop = config.dropout

    # Prepare base file name for results files
    kernel = '-kernel' if config.outer else ''
    act = '-' + config.act if config.act is not None else ''
    # alt_rnn = '2'
    n_glimpses = f'{config.n_glimpses}_' if config.n_glimpses is not None else ''
    detach = '-detach' if config.detach else ''
    pretrain = '-nopretrain' if config.no_pretrain else ''
    model_desc = f'{model_type}{detach}{act}{pretrain}_hsize-{config.h_size}_input-{train_on}{kernel}_{config.shape_input}'
    same = 'same' if config.same else ''
    challenge = config.challenge
    # solar = 'solarized_' if config.solarize else ''
    transform = 'logpolar_' if config.shape_input == 'logpolar' else 'gw6_'
    shapes = ''.join([str(i) for i in config.shapestr])
    sort = 'sort_' if config.sort else ''
    policy = config.policy

    data_desc = f'num{min_num}-{max_num}_nl-{noise_level}_grid{config.grid}_policy-{policy}_trainshapes-{shapes}{same}_{challenge}_{transform}{n_glimpses}{train_size}'
    # train_desc = f'loss-{use_loss}_niters-{n_iters}_{n_epochs}eps'
    withshape = '+shape' if config.learn_shape else ''
    train_desc = f'loss-{use_loss}{withshape}_opt-{config.opt}_drop{drop}_{sort}count-{target_type}_{n_epochs}eps_rep{config.rep}'
    base_name = f'{model_desc}_{data_desc}_{train_desc}'
    # if config.small_weights:
    #     base_name += '_small'
    return base_name


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--model_type', type=str, default='num_as_mapsum', help='rnn_classifier rnn_regression num_as_mapsum cnn')
    parser.add_argument('--target_type', type=str, default='multi', help='all or notA ')
    parser.add_argument('--train_on', type=str, default='xy', help='xy, shape, or both')
    parser.add_argument('--noise_level', type=float, default=1.6)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--grid', type=int, default=6)
    parser.add_argument('--n_iters', type=int, default=1, help='how many times the rnn should loop through sequence')
    parser.add_argument('--rotate', action='store_true', default=False)  # not implemented
    # parser.add_argument('--small_weights', action='store_true', default=False)  # not implemented
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--use_loss', type=str, default='both', help='num, map or both')
    parser.add_argument('--ventral', type=str, default=None)  
    parser.add_argument('--outer', action='store_true', default=False)
    parser.add_argument('--h_size', type=int, default=25)
    parser.add_argument('--min_pass', type=int, default=0)
    parser.add_argument('--max_pass', type=int, default=6)
    parser.add_argument('--min_num', type=int, default=2)
    parser.add_argument('--max_num', type=int, default=7)
    parser.add_argument('--act', type=str, default=None)
    parser.add_argument('--alt_rnn', action='store_true', default=False)
    # parser.add_argument('--no_symbol', action='store_true', default=False)
    parser.add_argument('--train_shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8], help='Can either be a string of numerals 0123 or letters ABCD.')
    parser.add_argument('--test_shapes', nargs='*', type=list, default=[[0, 1, 2, 3, 5, 6, 7, 8], [4]])
    parser.add_argument('--detach', action='store_true', default=False)
    parser.add_argument('--learn_shape', action='store_true', default=False, help='for the parametric shape rep, whether to additional train to produce symbolic shape labels')
    parser.add_argument('--shape_input', type=str, default='symbolic', help='Which format to use for what pathway (symbolic, parametric, tetris, char, noise, logpolar)')
    parser.add_argument('--same', action='store_true', default=False)
    parser.add_argument('--challenge', type=str, default='')
    parser.add_argument('--no_solarize', action='store_true', default=False)
    parser.add_argument('--n_glimpses', type=int, default=None)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--opt', type=str, default='SGD')
    # parser.add_argument('--tetris', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    # parser.add_argument('--preglimpsed', type=str, default=None)
    # parser.add_argument('--use_schedule', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--drop_rnn', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0) # 1e-6
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_act', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False, help='whether ventral stream was trained on sorted shape labels')
    parser.add_argument('--no_pretrain', action='store_true', default=False)
    parser.add_argument('--whole_image', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--policy', type=str, default='cheat+jitter')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    config = parser.parse_args()
    config.solarize = False if config.no_solarize else True
    if config.model_type == 'rnn_regression':
        config.cross_entropy = False
    else:
        config.cross_entropy = True
    # Convert string input argument into a list of indices
    if config.train_shapes[0].isnumeric():
        config.shapestr = config.train_shapes.copy()
        config.testshapestr = config.test_shapes.copy()
        config.train_shapes = [int(i) for i in config.train_shapes]
        for j, test_set in enumerate(config.test_shapes):
            config.test_shapes[j] = [int(i) for i in test_set]
    elif config.train_shapes[0].isalpha():
        letter_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
                      'J':8, 'K':9, 'N':10, 'O':11, 'P':12, 'R':13, 'S':14,
                      'U':15, 'Z':16}
        config.shapestr = config.train_shapes.copy()
        config.testshapestr = config.test_shapes.copy()
        config.train_shapes = [letter_map[i] for i in config.train_shapes]
        for j, test_set in enumerate(config.test_shapes):
            config.test_shapes[j] = [letter_map[i] for i in test_set]
    if 'ventral' in config.model_type and config.no_pretrain:
        assert 'finetune' in config.model_type  # otherwise the params in the ventral module will never be trained!
    print(config)
    return config
