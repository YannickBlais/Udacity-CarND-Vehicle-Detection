from lesson_functions import *
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm, grid_search, datasets
import pickle
from pathlib import Path
from sklearn.externals import joblib


training_folder = "./training_data"
vehicles_folder = training_folder + '/vehicles'
non_vehicles_folder = training_folder + '/non-vehicles'


def load_all_image_files(root):
  all_files = []
  for root, subFolders, files in os.walk(root):
    for filename in files:
      if filename.endswith(".png") or filename.endswith(".jpg") or \
        filename.endswith(".jpeg") or filename.endswith(".pgm"):

        all_files = all_files + [root + '/' + filename]

  return all_files


def make_terrain_data(cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                      pix_per_cell=8, cell_per_block=2, hog_channel=0,
                      spatial_feat=True, hist_feat=True, hog_feat=True):

  cars = load_all_image_files(vehicles_folder)
  not_cars = load_all_image_files(non_vehicles_folder)

  car_features = extract_features(cars[0:500], cspace, spatial_size, hist_bins,
                                  orient, pix_per_cell, cell_per_block, hog_channel,
                                  spatial_feat, hist_feat, hog_feat)
  not_car_features = extract_features(not_cars[0:500], cspace, spatial_size, hist_bins,
                                      orient, pix_per_cell, cell_per_block, hog_channel,
                                      spatial_feat, hist_feat, hog_feat)

  car_labels = np.ones(len(car_features))
  not_car_labels = np.zeros(len(not_car_features))

  all_features = car_features + not_car_features
  all_labels = car_labels.tolist() + not_car_labels.tolist()

  return train_test_split(all_features, all_labels, test_size=0.2)


def create_Scaler(features):

  res = np.vstack(features).astype(np.float64)
  from sklearn.preprocessing import StandardScaler
  return StandardScaler().fit(res)


def train_model(cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                pix_per_cell=8, cell_per_block=2, hog_channel=0,
                spatial_feat=True, hist_feat=True, hog_feat=True):

  model_filename = './model.plk'
  model_file = Path(model_filename)

  features_train, features_test, labels_train, labels_test = \
    make_terrain_data(cspace, spatial_size, hist_bins, orient,
                      pix_per_cell, cell_per_block, hog_channel,
                      spatial_feat, hist_feat, hog_feat)

  X_scaler = create_Scaler(features_train)
  features_train = X_scaler.transform(np.vstack(features_train).astype(np.float64))
  features_test = X_scaler.transform(np.vstack(features_test).astype(np.float64))

  if not model_file.exists():
    # parameters = {'C': [1, 10]}
    parameters = {'C': [1, 10]}
    # svr = svm.SVC()
    svr = LinearSVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(features_train, labels_train)
    print("best params: {}".format(clf.best_params_))
    joblib.dump(clf, model_filename)
  else:
    clf = joblib.load(model_filename)

  pred = clf.predict(features_test)

  from sklearn.metrics import accuracy_score
  acc = accuracy_score(pred, labels_test)
  print('accuracy = {}'.format(acc))

  return clf, X_scaler