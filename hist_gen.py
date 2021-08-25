### This is a script to generate the characters for the data directory, which are then just copy/pasted into the mod.
import random
import os

from continent_gen import KINGDOM_SIZE_LIST, CENTER_SIZE_LIST, BORDER_SIZE_LIST

SRC_DIR = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Crusader Kings III\\game\\common"
DATA_DIR = "data"
random.seed(1945)

def date_parser(date):
    if date:
        return date.replace('M',str(random.randint(1,12))).replace('D',str(random.randint(1,28))).replace('Y',str(random.randint(0,9)))
    else:
        return None

def character(cid, name, religion, culture, bd, female=False, dynasty=None, dd=None, father=None, mother=None,
              spouse_date = None, spouse_id = None, trait_list=[], stats_dict={}):
    '''Return the history/character entry for this character.'''
    bd = date_parser(bd)
    dd = date_parser(dd)
    spouse_date = date_parser(spouse_date)  # This one is a little weird because it should be shared between spouses.
    buf = "{} = {{\n\tname=\"{}\"\n".format(cid, name)
    if female:
        buf += "\tfemale = yes\n"
    if dynasty:
        buf += "\tdynasty={}\n".format(dynasty)
    buf += "\treligion=\"{}\"\n\tculture=\"{}\"\n".format(religion, culture)
    for stat, val in stats_dict.items():
        buf += "\t{}={}\n".format(stat, val)
    for trait in trait_list:
        buf += "\ttrait=\"{}\"\n".format(trait)
    if father:
        buf += "\tfather={}\n".format(father)
    if mother:
        buf += "\tmother={}\n".format(mother)
    buf += "\t{} = {{\n\t\tbirth=\"{}\"\n\t}}\n".format(bd, bd)
    if spouse_date and spouse_id:
        buf += "\t{} = {{\n\t\tadd_spouse = {}\n\t}}\n".format(spouse_date, spouse_id)
    if dd:
        buf += "\t{} = {{\n\t\tdeath=\"{}\"\n\t}}\n".format(dd, dd)
    buf += '}\n'
    return buf

def title_history(name, events, title_data, coffset):
    '''Return the history for the title.'''
    buf = f"{name} = {{\n"
    for k, v in events.items():
        if "." in k:
            event_date = date_parser(k)
            if isinstance(v[1], int):
                cid = v[1] + coffset
                buf += f"\t{event_date} = {{\n\t\t{v[0]}{cid}\n\t}}\n"
            elif "_" in v[1]:
                other_title = title_data[v[1]]
                buf += f"\t{event_date} = {{\n\t\t{v[0]}{other_title}\n\t}}\n"
        elif k == "development_level":
            buf += f"\t1000.1.1 = {{\tchange_development_level = {str(v)} }}\n"
        else:
            print(f"Unknown event: {k} {v}")
    if len(events) == 0 or name[0] == 'k':
        buf += "\t900.1.1 = {\n\t}\n"
    buf += "}\n"
    return buf

# Overall split of the kingdom is:
# k_title: old dead person maybe?
#   d_0 (5 counties): player
#     c_0 (6): player
#     c_1 (4): player
#     c_2 (4): player
#     c_3 (4): minor2  # Moved 1 to the other duchy for reasons.
#     c_4 (4): minor3
#   d_1 (4 counties): rival
#     c_5 (4): rival
#     c_6 (4): rival
#     c_7 (4): minor1 (liege: d_1)
#     c_8 (4): minor4 (liege: d_1)
#   d_2 (4 counties): empty
#     c_9 (4): minor5
#     c_10 (4): minor6
#     c_11 (3): minor5
#     c_12 (3): minor6
#   d_3 (3 counties): empty
#     c_13 (4): minor7
#     c_14 (4): minor8
#     c_15 (3): minor9

KINGDOM_DYNASTIES = {
    'd_0': 0,
    'd_1': 1,
    'c_7': 2,
    'c_3': 3,
    'c_4': 4,
    'c_8': 5,
    'c_9': 6,
    'c_10': 7,
    'c_13': 8,
    'c_14': 9,
    'c_15': 10,
}

KINGDOM_TITLES = {
    'k_0': {"development_level": 6},
    'd_0': {"development_level": 7, "980.1.1": ("holder = ", 0), "1000.1.1": ("holder = ", 2), },
    'c_0': {"980.1.1": ("holder = ", 0), "1000.1.1": ("holder = ", 2), },
    'c_1': {"980.1.1": ("holder = ", 0), "1000.1.1": ("holder = ", 2), },
    'c_2': {"980.1.1": ("holder = ", 0), "1000.1.1": ("holder = ", 2), },
    'c_3': {"98Y.1.1": ("holder = ", 30)},
    'c_4': {"98Y.1.1": ("holder = ", 35)},
    'd_1': {"development_level": 7, "985.1.1": ("holder = ", 10)},
    'c_5': {"985.1.1": ("holder = ", 10)},
    'c_6': {"990.1.1": ("holder = ", 10)},
    'c_7': {"990.1.1": ("holder = ", 21), "990.1.2": ("liege = ", "d_1")},
    'c_8': {"990.1.1": ("holder = ", 40), "990.1.2": ("liege = ", "d_1")},
    'd_2': {},
    'c_9': {"985.1.1": ("holder = ", 45)},
    'c_10': {"985.1.1": ("holder = ", 50)},
    'c_11': {"985.1.1": ("holder = ", 45)},
    'c_12': {"985.1.1": ("holder = ", 50)},
    'd_3': {},
    'c_13': {"985.1.1": ("holder = ", 55)},
    'c_14': {"985.1.1": ("holder = ", 60)},
    'c_15': {"985.1.1": ("holder = ", 65)},
}

KINGDOM_BARONIES = {
	'castle_holding': set([0, 6, 10, 14, 18, 22, 26, 30, 33, 36, 40, 44, 47, 50, 54, 58]),
	'church_holding': set([1, 7, 11, 15, 19, 23, 27, 31, 34, 37, 41, 45, 48, 51, 55, 59]),
	'city_holding': set([2, 8, 12, 16, 20, 24, 28, 32, 35, 38, 42, 46, 49, 52, 56, 60]),
}


KINGDOM_CHARS = {
    'player_father': {
        'cid': 0,
        'dynasty': 0,
        'bd': "940.M.D",
        'dd': "999.12.31",
        'spouse_id': 1,
        'spouse_date': "975.1.1",
    },
    'player_mother': {
        'cid': 1,
        'bd': "948.M.D",
        'female': True,
        'spouse_id': 0,
        'spouse_date': "975.1.1",
        'trait_list': ["education_intrigue_3"],
        'stats_dict': {"intrigue": 6},
    },
    'player': {
        'cid': 2,
        'dynasty': 0,
        'bd': "980.M.D",
        'father': 0,
        'mother': 1,
        'trait_list': ["education_martial_4", "ambitious", "blademaster_1"],
        'stats_dict': {"martial": 5, "diplomacy": 5, "intrigue": 5, "stewardship": 5, "learning": 5},
    },
    'player_brother': {
        'cid': 3,
        'dynasty': 0,
        'bd': "982.M.D",
        'father': 0,
        'mother': 1,
        'trait_list': ["education_stewardship_3", "content"],
        'stats_dict': {"martial": 3, "diplomacy": 3, "intrigue": 3, "stewardship": 3, "learning": 3},
    },
    'player_sister': {
        'cid': 4,
        'dynasty': 0,
        'female': True,
        'bd': "983.M.D",
        'father': 0,
        'mother': 1,
        'trait_list': ["education_diplomacy_3", "gregarious"],
        'stats_dict': {"martial": 3, "diplomacy": 3, "intrigue": 3, "stewardship": 3, "learning": 3},
    },

    'rival': {
        'cid': 10,
        'dynasty': 1,
        'bd': "96Y.M.D",
        'spouse_id': 22,
        'spouse_date': "985.1.1",
        'trait_list': ["physique_good_1", "shrewd"],
    },

    'minor1_father': {
        'cid': 20,
        'bd': "94Y.M.D",
        'dd': "990.1.1",
        'dynasty': 4,
    },
    'minor1': {
        'cid': 21,
        'bd': "96Y.M.D",
        'dynasty': 4,
        'father': 20,
    },
    'rival_wife': {
        'cid': 22,
        'bd': "96Y.M.D",
        'female': True,
        'dynasty': 4,
        'father': 20,
        'spouse_id': 10,
        'spouse_date': "985.1.1",
    },
}
KINGDOM_CHARS.update({f"minor{i}": {'cid': 20 + i * 5, 'dynasty': i + 1, 'bd': "97Y.M.D"} for i in range(2, 10)})

# Overall split of the center is:
#   d_0 (4 counties): main
#     c_0 (7): main
#     c_1 (5): main
#     c_2 (5): minor1
#     c_3 (5): minor2

CENTER_DYNASTIES = {
    'd_0': 0,
    'c_2': 1,
    'c_3': 2,
}

CENTER_TITLES = {
    'd_0': {"development_level": 8, "987.1.1": ("holder = ", 0),},
    'c_0': {"987.1.1": ("holder = ", 0)},
    'c_1': {"987.1.1": ("holder = ", 0)},
    'c_2': {"98Y.1.1": ("holder = ", 1)},
    'c_3': {"99Y.1.1": ("holder = ", 2)},
}

CENTER_BARONIES = {
    'castle_holding': set([0, 3, 7, 12, 17]),
	'church_holding': set([1, 4, 8, 13, 18]),
	'city_holding': set([2, 5, 9, 14, 19]),
}

CENTER_CHARS = {
    'main': {
        'cid': 0,
        'dynasty': 0,
        'bd': "970.M.D",
        'trait_list': ["education_martial_4", "intellect_good_1"],
        'stats_dict': {"martial": 4, "diplomacy": 4, "intrigue": 4, "stewardship": 4, "learning": 4},
    },
    'minor1': {
        'cid': 1,
        'bd': "96Y.M.D",
        'dynasty': 1,
    },
    'minor2': {
        'cid': 2,
        'bd': "97Y.M.D",
        'dynasty': 2,
    },
}


# Overall split of the border is:
#   d_0 (3 counties): empty
#     c_0 (4): minor1
#     c_1 (4): minor2
#     c_2 (4): minor3
#   d_1 (3 counties): empty
#     c_3 (4): minor4
#     c_4 (4): minor5
#     c_5 (4): minor6

BORDER_DYNASTIES = { f"c_{i}": i for i in range(6)}

BORDER_TITLES = {'d_0': {"development_level": 5}, 'd_1': {"development_level": 5},}
BORDER_TITLES.update({f"c_{i}": {"99Y.1.1": ("holder = ", i)} for i in range(6)})

BORDER_BARONIES = {
    'castle_holding': set([0, 4, 8, 12, 16, 20]),
	'church_holding': set([1, 9, 17]),
	'city_holding': set([6, 14, 22]),
}

BORDER_CHARS = {
    f"minor{i+1}": {
        'cid': i,
        'bd': "9YY.M.D",
        'dynasty': i,
    } for i in range(6)
}

culrel_map = {
 'b_alsace': ("swabian", "catholic"),
 'b_amdo': ("tuyuhun", "lamaism"),
 'b_bahrain': ("bedouin", "ashari"),
 'b_bohemia': ("czech", "catholic"),
 'b_capua': ("sicilian", "catholic"),
 'b_dzungaria': ("uyghur", "vajrayana"),
 'b_east_franconia': ("franconian", "catholic"),
 'b_genoa': ("cisalpine", "catholic"),
 'b_kham': ("bodpa", "lamaism"),
 'b_najd': ("bedouin", "ashari"),
 'b_shammar': ("bedouin", "ashari"),
 'b_silesia': ("polish", "catholic"),
 'b_toscana': ("italian", "catholic"),
 'b_west_franconia': ("franconian", "catholic"),
 'c_latium': ("italian", "catholic"),
 'c_lombardia': ("cisalpine", "catholic"),
 'c_mecca': ("bedouin", "ashari"),
 'c_prussia': ("prussian", "catholic"),
 'c_kiev': ("russian", "nestorian"),
#  'c_jerusalem': ("butr", "ismaili"),
 'e_eeurope': ("franconian", "catholic"),
 'e_islam': ("bedouin", "ashari"),
 'e_europe': ("italian", "catholic"),
 'k_aquitaine': ("occitan", "catholic"),
 'k_bavaria': ("bavarian", "catholic"),
 'k_burgundy': ("french", "catholic"),
 'k_east_francia': ("franconian", "catholic"),
 'k_egypt': ("egyptian", "mutazila"),
 'k_france': ("french", "catholic"),
 'k_hungary': ("hungarian", "catholic"),
 'k_italy': ("cisalpine", "catholic"),
 'k_norway': ("norse", "norse_pagan"),
 'k_pagan': ("burmese", "theravada"),
 'k_pomerania': ("pommeranian", "catholic"),
 'k_syria': ("levantine", "ashari"),
 'k_yemen': ("yemeni", "ashari"),
 'k_xia': ("han", "mahayana"),
 'r_catholic': ("italian", "catholic"),
 'r_islam': ("bedouin", "ashari"),
}

used_cultures = set()

culture_dict = {}
for culgroup in os.listdir(os.path.join(SRC_DIR, "culture", "cultures")):
    if culgroup[-1] != 't':
        continue
    with open(os.path.join(SRC_DIR, "culture", "cultures", culgroup), encoding='utf8') as inf:
        print(culgroup)
        name = None
        culinfo = {}
        names_mode = False
        names_mode = False
        names_name = None
        dynasty_names_mode = False
        used = False
        for line in inf.readlines():
            line = line.split("#")[0]
            if len(line) < 2:
                continue
            if names_mode:
                if '}' in line:
                    cul_info[names_name] = name_list
                    names_mode = False
                else:
                    if "\"" in line:
                        sline = line.split("\"")
                        for ind, group in enumerate(sline):
                            if ind % 2 == 1:
                                name_list.append(group)
                            else:
                                name_list.extend(group.strip().split())
                    else:
                        name_list.extend(line.strip().split())
            elif dynasty_names_mode:
                if line.startswith("\t\t}"):
                    cul_info["dynasty_names"] = name_list
                    dynasty_names_mode = False
                else:
                    name_list.append(line.strip())
            if line[1] != "\t" and "{" in line and "=" in line:
                maybe_name = line.split('=')[0].strip()
                if maybe_name[0]=='\ufeff':
                    maybe_name = maybe_name[1:]
                if maybe_name not in ["graphical_cultures", "mercenary_names"]:
                    if name:
                        culture_dict[name] = cul_info
                        print(name)
                    name = maybe_name
                    cul_info = {}
            if line.startswith("\t\tmale_names") or line.startswith("\t\tmale_Names"):  # fixing a bug with yemeni
                names_mode = True
                names_name = "male_names"
                name_list = []
            elif line.startswith("\t\tfemale_names") or line.startswith("\t\tfemale_Names"):
                names_mode = True
                names_name = "female_names"
                name_list = []
            elif line.startswith("\t\tdynasty_names"):
                dynasty_names_mode = True
                name_list = []
            elif line.startswith("\t\tdynasty_of_location_prefix"):
                cul_info["dynasty_of_location_prefix"] = line.split('=')[1].strip()
        culture_dict[name] = cul_info
        print(name)

# Coats of arms

coa_dict = {}
shortcut_dict = {}
with open(os.path.join(SRC_DIR, 'coat_of_arms', 'coat_of_arms', '01_landed_titles.txt'),'r', encoding='UTF8') as infile:
    coa_buffer = ""
    title_name = ""
    awake = False
    for line in infile:
        if len(line) > 1 and (line[0] == '@' or line[1] == '@'):
            if line[0] == '\ufeff':
                line = line[1:]
            shortcut_name = line[0:line.find(' ')]
            shortcut_dict[shortcut_name] = line
        if line[0] != '\t' and "=" in line and "{" in line and awake == False:
            title_name = line.split("=")[0].strip()
            coa_buffer = line
            awake = True
        elif awake and line[0] == "}":
            coa_buffer += line
            coa_dict[title_name] = coa_buffer
            title_name = ""
            awake = False
        elif awake:
            coa_buffer += line

# TODO: COA for dynasties.

# Process all of the titles in culrel_map.

for index, (title, (culture, religion)) in enumerate(culrel_map.items()):
    used_cultures.add(culture)
    if title[0] in ['e', 'r']:
        continue
    # Collect our title data.
    title_data = {}
    with open(os.path.join(DATA_DIR, title, "common","landed_titles", "landed_titles.txt"), encoding='utf8') as inf:
        d_count = 0
        c_count = 0
        for line in inf.readlines():
            if line.startswith("\tk_"):
                title_name = line.split("=")[0].strip()
                title_data[f"k_0"] = title_name
            if line.startswith("\t\td_"):
                title_name = line.split("=")[0].strip()
                title_data[f"d_{d_count}"] = title_name
                d_count += 1
            elif line.startswith("\t\t\tc_"):
                title_name = line.split("=")[0].strip()
                title_data[f"c_{c_count}"] = title_name
                c_count += 1

    coffset = (index + 1) * 100
    doffset = (index + 1) * 20
    if title[0] == 'b':
        dyn_todo, char_todo, title_todo = BORDER_DYNASTIES, BORDER_CHARS, BORDER_TITLES
    elif title[0] == 'c':
        dyn_todo, char_todo, title_todo = CENTER_DYNASTIES, CENTER_CHARS, CENTER_TITLES
    elif title[0] == 'k':
        dyn_todo, char_todo, title_todo = KINGDOM_DYNASTIES, KINGDOM_CHARS, KINGDOM_TITLES
    else:
        continue
    random.shuffle(culture_dict[culture]["dynasty_names"])
    os.makedirs(os.path.join(DATA_DIR, title, "common", "dynasties"), exist_ok=True)
    with open(os.path.join(DATA_DIR, title, "common","dynasties", "00_dynasties.txt"), 'w', encoding='utf8') as outf:
        for dyn, did in dyn_todo.items():
            if False:  # For now let's just use the prefix and the location.
                if len(culture_dict[culture]["dynasty_names"]) > 0:
                    dyn_name = culture_dict[culture]["dynasty_names"].pop()
                    if "{" in dyn_name:
                        processed_name = dyn_name.split("{")[1].split("}")[0].strip().split(" ")
                        if len(processed_name) == 2:
                            prefix, name = processed_name
                        elif len(processed_name) == 1:
                            prefix = None
                            name = processed_name[0]
                        else:
                            prefix = culture_dict[culture]["dynasty_of_location_prefix"]
                            name = "ERROR"
                    else:
                        prefix = None
                        name = dyn_name.strip()
            else:
                if "dynasty_of_location_prefix" in culture_dict[culture]:
                    prefix = culture_dict[culture]["dynasty_of_location_prefix"]
                else:
                    prefix = None
                if dyn in title_data:
                    name = title_data[dyn]
                else:
                    name = "ERROR"
            outf.write(f"{did + doffset} = {{\n")
            if prefix:
                outf.write(f"\tprefix = {prefix}\n")
            outf.write(f"\tname = \"{name}\"\n")
            outf.write(f"\tculture = \"{culture}\"\n}}\n")
    os.makedirs(os.path.join(DATA_DIR, title, "history", "characters"), exist_ok=True)
    with open(os.path.join(DATA_DIR, title, "history","characters", f"{title}.txt"), 'w', encoding='utf8') as outf:
        for char_dict in char_todo.values():
            others = {k: v + coffset if k in ['father', 'mother', 'spouse_id'] else v for k, v in char_dict.items() if k not in ["cid", "dynasty"]}
            cid = char_dict["cid"] + coffset
            dynasty = char_dict["dynasty"] + doffset if "dynasty" in char_dict else None
            try:
                name = random.choice(culture_dict[culture]["female_names" if "female" in others else "male_names"])
            except:
                name = "ERROR"
            outf.write(character(cid=cid, name=name, religion=religion, culture=culture, **others, dynasty=dynasty))
    os.makedirs(os.path.join(DATA_DIR, title, "history", "titles"), exist_ok=True)
    with open(os.path.join(DATA_DIR, title, "history", "titles", f"{title}.txt"), 'w', encoding='utf8') as outf:
        for t_short, ttd in title_todo.items():
            assert t_short in title_data, (str(t_short), str(title_data))
            outf.write(title_history(title_data[t_short], ttd, title_data, coffset))
    os.makedirs(os.path.join(DATA_DIR, title, "history", "provinces"), exist_ok=True)
    with open(os.path.join(DATA_DIR, title, "history", "provinces", f"{title}.txt"), 'w', encoding='utf8') as outf:
        bnum = 0
        if title[0] == 'k':
            castle_holding = KINGDOM_BARONIES['castle_holding']
            city_holding = KINGDOM_BARONIES['city_holding']
            church_holding = KINGDOM_BARONIES['church_holding']
            outf.write(f"#{title}\n")
            for dind, duchy in enumerate(KINGDOM_SIZE_LIST):
                dname = title_data["d_"+str(dind)]
                outf.write(f"##{dname}\n")
                for cind, county in enumerate(duchy):
                    cname = title_data["c_"+str(cind)]
                    outf.write(f"###{cname}\n")
                    for bind in range(county):
                        outf.write(f"P{bnum} = {{\n")
                        if bind == 0:
                            outf.write(f"\tculture = {culture}\n")
                            outf.write(f"\treligion = {religion}\n")
                        if bnum in castle_holding:
                            outf.write(f"\tholding = castle_holding\n")
                        elif bnum in city_holding:
                            outf.write(f"\tholding = city_holding\n")
                        elif bnum in church_holding:
                            outf.write(f"\tholding = church_holding\n")
                        else:
                            outf.write(f"\t#holding = none\n")
                        outf.write("}\n")
                        bnum += 1
                    outf.write("\n")
        elif title[0] == 'c':
            castle_holding = CENTER_BARONIES['castle_holding']
            city_holding = CENTER_BARONIES['city_holding']
            church_holding = CENTER_BARONIES['church_holding']
            outf.write(f"##{title}\n")
            for cind, county in enumerate(CENTER_SIZE_LIST):
                cname = title_data["c_"+str(cind)]
                outf.write(f"###{cname}\n")
                for bind in range(county):
                    outf.write(f"P{bnum} = {{\n")
                    if bind == 0:
                        outf.write(f"\tculture = {culture}\n")
                        outf.write(f"\treligion = {religion}\n")
                    if bnum in castle_holding:
                        outf.write(f"\tholding = castle_holding\n")
                    elif bnum in city_holding:
                        outf.write(f"\tholding = city_holding\n")
                    elif bnum in church_holding:
                        outf.write(f"\tholding = church_holding\n")
                    else:
                        outf.write(f"\t#holding = none\n")
                    outf.write("}\n")
                    bnum += 1
                outf.write("\n")
        elif title[0] == 'b':
            castle_holding = BORDER_BARONIES['castle_holding']
            city_holding = BORDER_BARONIES['city_holding']
            church_holding = BORDER_BARONIES['church_holding']
            for dind in range(2):
                dname = title_data["d_"+str(dind)]
                outf.write(f"##{dname}\n")
                for cind, county in enumerate(BORDER_SIZE_LIST):
                    cname = title_data["c_"+str(cind)]
                    outf.write(f"###{cname}\n")
                    for bind in range(county):
                        outf.write(f"P{bnum} = {{\n")
                        if bind == 0:
                            outf.write(f"\tculture = {culture}\n")
                            outf.write(f"\treligion = {religion}\n")
                        if bnum in castle_holding:
                            outf.write(f"\tholding = castle_holding\n")
                        elif bnum in city_holding:
                            outf.write(f"\tholding = city_holding\n")
                        elif bnum in church_holding:
                            outf.write(f"\tholding = church_holding\n")
                        else:
                            outf.write(f"\t#holding = none\n")
                        outf.write("}\n")
                        bnum += 1
                    outf.write("\n")
    with open(os.path.join(DATA_DIR, title, "common", "landed_titles", "landed_titles.txt"), encoding='utf8') as inf:
        buffer = ""
        for line in inf.readlines():
            if "=" in line and "{" in line and "_" in line:
                small_title = line.split("=")[0].strip()
                if small_title in coa_dict:
                    buffer += coa_dict[small_title]
        if len(buffer) > 0:
            out_dir = os.path.join(DATA_DIR, title, "common", "coat_of_arms", "coat_of_arms")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{title}.txt"), 'w', encoding='utf8') as outf:
                for shortcut, shortcut_text in shortcut_dict.items():
                    if shortcut in buffer:
                        outf.write(shortcut_text)
                outf.write(buffer)

# Culture history (innovations)
innovations = {
    "tribal": ["innovation_motte", "innovation_catapult", "innovation_barracks", "innovation_mustering_grounds", "innovation_bannus", "innovation_quilted_armor",
               "innovation_development_01", "innovation_currency_01", "innovation_gavelkind", "innovation_crop_rotation",
               "innovation_city_planning", "innovation_casus_belli", "innovation_plenary_assemblies", "innovation_ledger"],
    "early_medieval": ["innovation_battlements", "innovation_mangonel", "innovation_burhs", "innovation_house_soldiers", "innovation_horseshoes", "innovation_arched_saddle",
                       "innovation_manorialism", "innovation_development_02", "innovation_currency_02", "innovation_royal_prerogative", "innovation_armilary_sphere"],
    "guaranteed": ["innovation_chronicle_writing"],
    "banned": ["innovation_hereditary_rule", "innovation_baliffs"],
}

# The plan here is that all cultures start out with all tribal innovations, chronicle_writing (which will help unshatter the world),
# and three other random early medieval innovations.
os.makedirs(os.path.join(DATA_DIR, "start", "history", "cultures"), exist_ok=True)
for culture in culture_dict.keys():
    if culture.endswith("_group"):
        # Groups only get tribal innovations, so that you don't get multiple rolls.
        with open(os.path.join(DATA_DIR, "start", "history", "cultures", f"{culture}.txt"), 'w', encoding='utf8') as outf:
            outf.write(f"#{culture}\n\n900.1.1 = {{\n")
            outf.write("\n".join(f"\tdiscover_innovation = {i}" for i in innovations["tribal"]))
            outf.write("\n\tjoin_era = culture_era_early_medieval\n}\n")
    else:
        with open(os.path.join(DATA_DIR, "start", "history", "cultures", f"{culture}.txt"), 'w', encoding='utf8') as outf:
            outf.write(f"#{culture}\n\n900.1.1 = {{\n")
            outf.write("\n".join(f"\tdiscover_innovation = {i}" for i in innovations["tribal"]))
            outf.write("\n\tjoin_era = culture_era_early_medieval\n}\n\n1000.1.1 = {\n")
            random.shuffle(innovations["early_medieval"])
            outf.write("\n".join(f"\tdiscover_innovation = {i}" for i in innovations["guaranteed"]))
            outf.write("\n")
            outf.write("\n".join(f"\tdiscover_innovation = {i}" for i in innovations["early_medieval"][:3]))
            outf.write("\n}\n")
