import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from lesson_functions import *
from training import *
from scipy.ndimage.measurements import label
import time


orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
cspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 0
debug_folder = './debug'
spatial_feat = False
hist_feat = True
hog_feat = True


print("Training model...")
svc, X_scaler = train_model(cspace, spatial_size, hist_bins, orient, pix_per_cell,
                            cell_per_block, hog_channel,
                            spatial_feat, hist_feat, hog_feat)
print("Training completed!")

# for root, subFolders, files in os.walk('./test_images'):
#   for filename in files:
#     if filename.endswith(".png") or filename.endswith(".jpg") or \
#       filename.endswith(".jpeg") or filename.endswith(".pgm"):
#       img = mpimg.imread('./test_images/' + filename)
#       # img = img * 255.0
#       working_img = img.copy()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_test_video.mp4', fourcc, 25.0, (1280, 720))
out2 = cv2.VideoWriter('output_bbox_test_video.mp4', fourcc, 25.0, (1280, 720))

cap = cv2.VideoCapture('./videos/project_video.mp4')
i = 0

start_debug = 0
end_debug = 1000000
last_frame_boxes_found = []
while (cap.isOpened()):
  ret, img = cap.read()
  if img is None:
    break
  i += 1
  if i < start_debug or i > end_debug:
    continue
  #
  # windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 668],
  #                        xy_window=(64, 64), xy_overlap=(0.1, 0.1))

  windows = slide_window(img, x_start_stop=[400, 1280], y_start_stop=[400, 668],
                         xy_window=(96, 96), xy_overlap=(0.7, 0.7))

  windows = windows + slide_window(img, x_start_stop=[400, 1280], y_start_stop=[375, 668],
                         xy_window=(128, 128), xy_overlap=(0.5, 0.5))

  windows = windows + slide_window(img, x_start_stop=[400, 1280], y_start_stop=[350, 668],
                         xy_window=(192, 192), xy_overlap=(0.5, 0.5))

  start = time.time()
  hot_windows = search_windows(img, windows, svc, X_scaler, color_space=cspace,
                               spatial_size=spatial_size, hist_bins=hist_bins,
                               orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block,
                               hog_channel=hog_channel,
                               spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
  end = time.time()
  print(end - start)

  img_boxes = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
  # cv2.imwrite(debug_folder + '/boxes.png', img_boxes)

  # plt.imshow(window_img)

  heat = np.zeros_like(img[:, :, 0]).astype(np.float)

  # Add heat to each box in box list
  heat = add_heat(heat, hot_windows)

  # Apply threshold to help remove false positives
  heat = apply_threshold(heat, 1)

  # Visualize the heatmap when displaying
  heatmap = np.clip(heat, 0, 255)

  # Find label boxes from heatmap using label function
  labels = label(heatmap)
  label_img = np.copy(img)
  label_img, label_boxes = draw_labeled_bboxes(label_img, labels)

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

  # cv2.imwrite(debug_folder + '/final_boxes.png', final_boxes)
  # cv2.imwrite(debug_folder + '/heat_map.png', heatmap)
  # fig = plt.figure()
  # plt.subplot(121)
  # plt.imshow(final_boxes)
  # plt.title('Car Positions')
  # plt.subplot(122)
  # plt.imshow(heatmap, cmap='hot')
  # plt.title('Heat Map')
  # fig.tight_layout()

  out.write(img)
  out2.write(img_boxes)