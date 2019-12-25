import pickle

with open("select_info", "rb") as f:
    select_info = pickle.load(f)
    select_times = dict()
    for epoch, d in select_info.items():
        for k, v in d.items():
            if k not in select_times:
                select_times[k] = 1
            else:
                select_times[k] += 1
    select_times = sorted(select_times.items(), key=lambda item: item[0])
    print(select_times)

