# easy_run_Phil.py
# edited from "easy_run.py" on 10/07/2019
# wouldn't work. last error:
#     Traceback (most recent call last):
#       File "recognition.py", line 584, in <module>
#         with open(paths['config']) as config_file:
#     FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\Phil\\SU-ECE-19-7-master\\config.json'

import os

builder = {'-image_source': '', '-template_source': '', '-config_source': '',
                    '-cluster_source': '', '-destination': '', '-num_threads': '',
                    '-write_threshold': ''}

builder['-image_source'] = str('C:/Users/Phil/SU-ECE-19-7-master/Image_Sets/quick_set/images/*')
builder['-template_source'] = str('C:/Users/Phil/SU-ECE-19-7-master/templates/*') 
builder['-config_source'] = str('"C:/Users/Phil/SU-ECE-19-7-master/Image_Sets/quick_set/config.json"')
builder['-cluster_source'] = str('"C:/Users/Phil/SU-ECE-19-7-master/Image_Sets/quick_set/cluster_table.csv"')
builder['-destination'] = str('C:/Users/Phil/SU-ECE-19-7-master/destination')
builder['-num_threads'] = 1
builder['-write_threshold'] = 30

command_line = str('python recognitionv2.py')
for argument in builder:
    if builder[argument] != None:
        next = str(' {0} "{1}"'.format(str(argument), str(builder[argument])))
        command_line = command_line + next

os.system(command_line)
