import os

builder = {'-image_source': '', '-template_source': '', '-config_source': '',
                    '-cluster_source': '', '-destination': '', '-num_threads': '',
                    '-write_threshold': ''}

builder['-image_source'] = str('/Users/tonycaballero/Downloads/SU-ECE-20-4-master/Image_Sets/quick_set/images/*')
builder['-template_source'] = str('/Users/tonycaballero/Downloads/SU-ECE-20-4-master/Image_Sets/quick_set/templates/*')
builder['-config_source'] = str('/Users/tonycaballero/Downloads/SU-ECE-20-4-master/Image_Sets/quick_set/config.json')
builder['-cluster_source'] = None
builder['-destination'] = str('/Users/tonycaballero/Downloads/SU-ECE-20-4-master/Image_Sets/quick_set/destination')
builder['-num_threads'] = 1
builder['-write_threshold']=30

command_line = str('python recognition.py')
for argument in builder:
    if builder[argument] != None:
        next = str(' {0} "{1}"'.format(str(argument), str(builder[argument])))
        command_line = command_line + next

os.system(command_line)
