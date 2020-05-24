import os

builder = {'-image_source': '', '-template_source': '', '-config_source': '',
                    '-cluster_source': '', '-destination': '', '-num_threads': '',
                    '-write_threshold': '', '-validation_dataset': '', '-weight_source': ''}

builder['-image_source'] = str('C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Image_Sets/set_1/images/*')
builder['-template_source'] = str('C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Image_Sets/set_1/templates/*')
builder['-config_source'] = str('C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Image_Sets/set_1/config.json')
builder['-cluster_source'] = None
builder['-destination'] = str('C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Image_Sets/set_1/destination')
builder['-num_threads'] = 4
builder['-write_threshold'] = 30
builder['-validation_dataset'] = str('C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Recognition/samples/snow_leopard/dataset')
builder['-weight_source'] = str('C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Recognition/Mask_RCNN-master/logs/bottle20200221T0110/mask_rcnn_bottle_0010.h5')

command_line = str('python recognition-MaskRCNN.py')
for argument in builder:
    if builder[argument] != None:
        next = str(' {0} "{1}"'.format(str(argument), str(builder[argument])))
        command_line = command_line + next

os.system(command_line)
