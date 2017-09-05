from training import *
from scipy.ndimage.measurements import label
import time


orient = 32
pix_per_cell = 16
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
cspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL'
debug_folder = './debug'
spatial_feat = True
hist_feat = True
hog_feat = True


print("Training model...")
svc, X_scaler = train_model(cspace, spatial_size, hist_bins, orient, pix_per_cell,
                            cell_per_block, hog_channel,
                            spatial_feat, hist_feat, hog_feat)
print("Training completed!")


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_test_video.mp4', fourcc, 25.0, (1280, 720))
out2 = cv2.VideoWriter('output_bbox_test_video.mp4', fourcc, 25.0, (1280, 720))

cap = cv2.VideoCapture('./videos/project_video.mp4')
i = 0

start_debug = 1000
end_debug = 1200
# start_debug = 0
# end_debug = 1000000
fig = None
row = 1
last_frame_boxes_found = []
while (cap.isOpened()):
  ret, img = cap.read()
  if img is None:
    break
  i += 1
  if i < start_debug or i > end_debug:
    continue

  start = time.time()
  hot_windows = []
  windows = find_cars(img, cspace, ystart=400, ystop=500, scale=0.5, svc=svc, X_scaler=X_scaler, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, hog_img=None)
  hot_windows += windows

  windows = find_cars(img, cspace, ystart=400, ystop=550, scale=0.75, svc=svc, X_scaler=X_scaler, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, hog_img=None)
  hot_windows += windows

  windows = find_cars(img, cspace, ystart=400, ystop=550, scale=1.0, svc=svc, X_scaler=X_scaler, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, hog_img=None)
  hot_windows += windows

  windows = find_cars(img, cspace, ystart=375, ystop=656, scale=2.0, svc=svc, X_scaler=X_scaler, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, hog_img=None)
  hot_windows += windows

  windows = find_cars(img, cspace, ystart=375, ystop=656, scale=3.0, svc=svc, X_scaler=X_scaler, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins, hog_img=None)
  hot_windows += windows

  end = time.time()
  print(end - start)

  img_boxes = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)

  # plt.imshow(window_img)

  heat = np.zeros_like(img[:, :, 0]).astype(np.float)

  # Add heat to each box in box list
  heat = add_heat(heat, hot_windows)

  # Apply threshold to help remove false positives
  heat = apply_threshold(heat, 0)

  # Visualize the heatmap when displaying
  heatmap = np.clip(heat, 0, 255)

  # Find label boxes from heatmap using label function
  labels = label(heatmap)
  label_img = np.copy(img)
  label_img, label_boxes = draw_labeled_bboxes(label_img, labels)
  fig = plt.figure()
  plt.imshow(labels[0], cmap='gray')

  # Filter boxes using boxes-found-on-last-frame
  final_windows = []
  for label_box in label_boxes:
    for last_window in last_frame_boxes_found:
      last_window_center = ((last_window[1][0] + last_window[0][0])/2.0, (last_window[1][1] + last_window[0][1])/2.0)
      if last_window_center[0] > label_box[0][0] and last_window_center[0] < label_box[1][0] and \
        last_window_center[1] > label_box[0][1] and last_window_center[1] < label_box[1][1]:
        final_windows.append((((last_window[0][0] + label_box[0][0]) / 2, (last_window[0][1] + label_box[0][1]) / 2),
                             ((last_window[1][0] + label_box[1][0]) / 2, (last_window[1][1] + label_box[1][1]) / 2)))

  last_frame_boxes_found = label_boxes

  for window in final_windows:
    cv2.rectangle(img, window[0], window[1], (0, 0, 255), 6)

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)

  # img_boxes = cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB)
  # if fig == None:
  #   fig = plt.figure()
  # plt.subplot(6, 2, row)
  # plt.imshow(img_boxes)
  # plt.title('Car Positions')
  # plt.subplot(6, 2, row + 1)
  # plt.imshow(heatmap, cmap='hot')
  # plt.title('Heat Map')
  # fig.tight_layout()
  # row += 2

  out.write(img)
  out2.write(img_boxes)