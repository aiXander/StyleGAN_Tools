import os, sys, random, cv2, warnings, time, pickle, shutil, collections
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append('/home/stylegan2')  #Add stylegan2 source dir to path
import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib


'''

ToDo:
- add img_path renaming
- add option to go backwards

'''

network_pkl1 = '/home/stylegan2/results/00037-stylegan2-micro-1gpu-cond-config-f/network-snapshot-000096.pkl'
network_pkl2 = '/home/stylegan2/results/00038-stylegan2-micro-1gpu-cond-config-f/network-snapshot-000145.pkl'

label_size = 6
n = 200
latents = None
#latents = 'cherries/z_dict_TATE_FINAL.pkl'

trunc_value = 1.0
batch_size = 8
_N = 512
img_res = (1024,1024,3)

dump_images    = True
compare_images = True

if not dump_images:
    dump_img_dir = 'model_compare_images/1585217910.462612/'
else:
    dump_img_dir = 'model_compare_images/%s' %str(time.time())
    

def get_latents(label_size, n, N=512):
    type_choices = [0,1,2,3]
    #type_choices = [2]

    z = np.empty((n,N))
    l = np.empty((n,label_size))
    
    for i in range(n):
        latent_type = random.choice(type_choices)
        if label_size > 0:
            if latent_type == 0:
                l1 = np.random.randn(1,label_size)
            if latent_type == 1: 
                l1 = np.random.uniform(size=(1,label_size))
                l1 = l1 / np.sum(l1)
            if latent_type == 2: 
                l1 = np.zeros((1,label_size))
                l1[:,random.choice(list(range(label_size)))] = 1
            if latent_type == 3:
                l1 = np.zeros((1,label_size))
            
            l[i] = l1
        z[i] = np.random.randn(1,N)

    return z, l


def get_w_from_z(Gs_network, z, conditioning = None, compact = False, N=512):
    bs = z.shape[0]
    if z.shape[1] > N:
        conditioning = z[:,N:].copy().reshape(bs,-1)
        z = z[:,:N].copy().reshape(bs,N)

    if not compact:
        return Gs_network.components.mapping.run(z, conditioning, minibatch_size=z.shape[0])
    else:
        return Gs_network.components.mapping.run(z, conditioning, minibatch_size=z.shape[0])[:,0,:]


def dump_imgs(model_pkl_path, target_dir, latent_vector, truncation_psi, z=True, noise=False, N=_N, w_dim=18):

    _, _, Gs_network = pretrained_networks.load_networks(model_pkl_path)
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=batch_size)
    dlatent_avg = Gs_network.get_var('dlatent_avg')

    bs = latent_vector.shape[0]

    if latent_vector.shape[1] > N:
      latent_vector = latent_vector[:,:N].copy()
      conditioning = latent_vector[:,N:].copy()

    if z:
        latent_vector = latent_vector.reshape((bs, 512))
        w = get_w_from_z(Gs_network, latent_vector)
    else: 
        w = latent_vector.reshape((bs, w_dim, 512))

    if truncation_psi is not None:
      if truncation_psi != 1:
        w[:, :cutoff_layer, :] = truncation_psi*w[:, :cutoff_layer, :] + (1-truncation_psi)*dlatent_avg

    print("Generating %d images for model %s" %(len(latent_vector), model_pkl_path.split('/')[-1]))
    img_array = Gs_network.components.synthesis.run(w, randomize_noise=noise, **synthesis_kwargs)

    print('Generated image array with shape:')
    print(img_array.shape)
    
    print("Saving imgs to disk...")
    os.makedirs(target_dir, exist_ok = False)
    img_paths = []
    for i, img in enumerate(img_array):
        img_path = target_dir + 'img_%05d_unlabeled.jpg' %i
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_paths.append(img_path)

    return img_paths

if dump_images:
    os.makedirs(dump_img_dir, exist_ok = False)

    if latents is None:
        z,l = get_latents(label_size, n)
        latent_walk = np.concatenate((z,l), axis=1)
    else:
        latents = pickle.load( open(latents, "rb" ) )
        latent_z = []
        for label in sorted(latents.keys()):
            latent_z.append(latents[label])

        latent_walk = np.vstack(latent_z)
        latent_walk = latent_walk[order]
        

    latent_walk = latent_walk[:n]
    print("Latent walk shape:")
    print(latent_walk.shape)
    np.save("generated_z.npy", latent_walk)

    img_paths1 = sorted(dump_imgs(network_pkl1, dump_img_dir + '/model_1/', latent_walk, trunc_value))
    img_paths2 = sorted(dump_imgs(network_pkl2, dump_img_dir + '/model_2/', latent_walk, trunc_value))

else:
    img_paths1 = sorted([dump_img_dir + '/model_1/' + f for f in os.listdir(dump_img_dir + '/model_1/')])
    img_paths2 = sorted([dump_img_dir + '/model_2/' + f for f in os.listdir(dump_img_dir + '/model_2/')])

assert len(img_paths1) == len(img_paths2)

###################################################################
###################################################################
#Compare the generated images visually:
###################################################################
###################################################################

if compare_images:

    buffer_size   = 20
    file_buffer = collections.deque(maxlen=buffer_size)

    window_size = 2
    f = 0.90
    cv2.namedWindow('display')
    pos = 10
    cv2.moveWindow('display', pos, pos)
    background = np.zeros((img_res[0], img_res[1]*2, 3))

    global canvas
    canvas = background.copy()

    def display_with_labels(background, image1, image2, cnt, n):
        global canvas
        canvas = background.copy()
        canvas[:, :img_res[1] , :] = image1/255
        canvas[:, img_res[1]: , :] = image2/255
        canvas = add_progress_bar(canvas, cnt, n)

        canvas = cv2.resize(canvas, (int(canvas.shape[1]*f), int(canvas.shape[0]*f)))  
        cv2.imshow("display", canvas)
        return

    def add_progress_bar(canvas, cnt, n, bar_w = 25):
        bar = np.zeros((bar_w, canvas.shape[1], 3))
        fill_x = int(canvas.shape[1]*cnt/n)
        bar[:,:fill_x,:] = np.array([255,50,50])
        canvas[-bar_w:, :, :] = bar
        return canvas

    def label_image(img_path1, img_path2, step, n):

        if bool(random.getrandbits(1)):
            image1 = cv2.imread(img_path1)
            image2 = cv2.imread(img_path2)
            left_label = 'model1'
            right_label = 'model2'
        else:
            image1 = cv2.imread(img_path2)
            image2 = cv2.imread(img_path1)
            left_label = 'model2'
            right_label = 'model1'

        display_with_labels(background, image1, image2, step, n)

        while True:
            k = cv2.waitKey(0)
            if k==27: break # Esc key to skip this image
            elif k==-1: continue #Default when no key is pressed
            else:
                if (k>48) and (k<58): #1,2,3,4,5,6,7,8,9   49,50,51,52,53,54,55,56,57
                    return -1, 'idk'
                elif k==107: #k
                    return 1, left_label
                elif k==108: #l
                    return 1, right_label
                elif k==13:  #Enter --> no preference
                    return 1, 'idk'


    n = len(img_paths1)
    print("Found 2x %d images" %n)

    labels, cnt = [], 0

    while cnt<(n-1):
        step, label = label_image(img_paths1[cnt], img_paths2[cnt], cnt, n)

        if step == -1:
            del labels[-1]

        if label == 'model1':
            labels.append(0)
        if label == 'model2':
            labels.append(1)

        cnt = cnt + step
        cnt = max(0,cnt)

        if cnt%10 == 0:
            try:
                score = 100*np.mean(labels)
                sys.stdout.write("\r %d of %d imgs (%.1f%%) -- Current score: %.1f%% for model1 vs %.1f%% for model2)\n" %(cnt, n, 100*cnt/n, 100-score, score))
            except:
                pass
    print(labels)
    print("\nAvg score: %.2f" %np.mean(labels))

    if np.mean(labels) > 0.5:
        winning_model = 'model2'
        winning_pkl = network_pkl2
        pref = 100*np.mean(labels)
    else:
        winning_model = 'model1'
        winning_pkl = network_pkl1
        pref = 100 - 100*np.mean(labels)

    print("Model %s wins with a preference of %.2f%%" %(winning_model, pref))
    print("You should use pkl:")
    print("\n%s" %winning_pkl)