
import os

def check_landed_titles():
    '''Check to ensure that all files in data:
        - have all the correct the barony numnbers assigned
        - have equal numbers of { and } [but they might be in the wrong places]
        - don't have any extra baronies.
    Some failures will cause this to crash, which will hopefully still be informative.'''
    problems = False
    for title_name in os.listdir('data\\'):
        out_file = os.path.join('data\\', title_name, "common", "landed_titles", "landed_titles.txt")
        if title_name[0] == 'k':
            present = [False] * 61
        elif title_name[0] == 'b':
            present = [False] * 24
        elif title_name[0] == 'c':
            present = [False] * 22
        else:
            continue
        with open(out_file, 'r', encoding='UTF8') as infile:
            left_brace = 0
            right_brace = 0
            extra = 0
            for line in infile.readlines():
                if '{' in line:
                    left_brace += 1
                if '}' in line:
                    right_brace += 1
                if "province =" in line:
                    pid = line.split('=')[1].strip()[1:]
                    if len(pid) == 0:
                        extra += 1
                    else:
                        present[int(pid)] = True
        if all(present) and extra == 0 and left_brace == right_brace:
            print(title_name, "Correct!")
        else:
            print(title_name, "Error!")
            if not(all(present)):
                missing = []
                for pind, p in enumerate(present):
                    if not p:
                        missing.append(pind)
                print("\tmissing: ", missing)
            if extra != 0:
                print("\textra: ", extra)
            if left_brace != right_brace:
                print("\tleft: ", left_brace, "right: ", right_brace)

if __name__ == '__main__':
    check_landed_titles()