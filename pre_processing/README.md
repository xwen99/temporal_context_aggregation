# Data Preparation
## Getting the Dataset
* Download the dataset you want. The supported options are:
    * [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/)
    * [VCDB](http://www.yugangjiang.info/research/VCDB/index.html)
    * [FIVR-5K, FIVR-200K](http://ndd.iti.gr/fivr/)
    * [EVVE](http://pascal.inrialpes.fr/data/evve/)
* For convenience, please keep the folder structure as default.
## Extracting the Frames
* Extract the frames of all videos by a given fps, perform resize & crop, then save them to seperate .npy files.
* Please explore the toolbox in `extract_frames.py`.
* Example usage on the VCDB dataset:
```python
vid2paths_core, vid2paths_bg = load_video_paths_vcdb()
pool = Pool(4)
for vid, path in tqdm(vid2paths_core.items()):
    a = np.load('~/datasets/vcdb/frames/core/' + vid + '.npy')
    if a.shape[0] == 0 or not os.path.exists('~/datasets/vcdb/frames/core/' + vid + '.npy'):
        pool.apply_async(h, ((vid, path),))
pool.close()
pool.join()
```
## Extracting the Feature
* Extract the feature of video frames with a pre-trained backbone, then save them to sepereta .npy files.
* Please explore the toolbox in `extract_features.py`.
* Example usage on the VCDB dataset:
```python
v2v = Video2Vec(dataset='vcdb', model='resnet-50', num_workers=8)
v2v.get_vec(path='~/datasets/vcdb/')
```
## Applying PCA Whitening
* First, extract the feature list.
* Please explore the toolbox in `feature_post_processing.py`.
* Example usage on the VCDB dataset:
```python
feature_list = get_feature_list(dataset='vcdb', feat='imac')
export_feature_list(feature_list, out_path='vcdb_feature_paths_imac.txt')
```

* Then, train the PCA module with the VCDB dataset.
* Here, we randomly sample 10 frames per video.
* Example usage:
```python
def pipe(a):
    a = np.load(a)
    a = a[np.random.choice(len(a), 10), :]
    return a

feature_path='feature_paths/vcdb_feature_paths_imac.txt'
paths = [l.split('\t')[1].strip() for l in open(feature_path, 'r').readlines()]

pool = Pool(32)
features = []
for path in paths:
    features += [pool.apply_async(pipe,
                    args=[path],
                    callback=(lambda *a: progress_bar.update()))]
pool.close()
pool.join()

feat_array = []
for feat in tqdm(features):
    feat_array.append(feat.get())
feats = np.concatenate(feat_array)

pca = PCA(parameters_path='pca_params_vcdb997090_resnet50_imac_3840.npz')
pca.train(feats)
```
* The parameters of the PCA module is then saved to the `parameters_path`.

* Then, apply PCA on the features:
```python
pca = PCA(parameters_path='pca_params_vcdb997090_resnet50_imac_3840.npz')
pca.load()

def f(path):
    feat = np.load(path)
    feat = pca.infer(torch.Tensor(feat)).numpy()
    path = path.replace('imac', 'imac_pca1024')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(ps.path.diranme(path))
    np.save(path, feat)

paths = get_feature_list(dataset='vcdb', feat='imac')

pool = Pool()
for path in tqdm(paths):
    if not os.path.exists(path.replace('imac', 'imac_pca1024')):
        pool.apply_async(f, args=(path,))
pool.close()
pool.join()
```
## The Final Step
* Finally, export the dataset to a h5py file for convenience (optional):
* See the `npy2h5py` function in `feature_post_processing.py`.
* Example usage:
```python
feature_path = 'feature_paths/vcdb_resnet50_imac_pca_1024.txt'
h5path = 'vcdb_imac.hdf5'
npy2h5py(feature_list_path, h5path, pca=None)
```