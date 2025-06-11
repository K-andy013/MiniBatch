import Augmentor

p = Augmentor.Pipeline("D:/Hafsa/woundSegmentation/augmented-data/train_aug/images")
p.ground_truth("D:/Hafsa/woundSegmentation/augmented-data/train_aug/labels")
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.set_save_format(save_format='auto')
# p.set_save_directory("")
p.sample(5000, multi_threaded=False)