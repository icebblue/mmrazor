
logfile = './work_dirs/js-div_logits_i3d_top-i3d_4xb4-64x1x1_ucf101/20240124_130842/20240124_130842.log'

with open(logfile, "r") as f:
    cnt = 0
    sum = 0.0
    for line in f:
        if "distill.loss_kl:" in line:
            strs = line.split(' ')
            cnt += 1
            sum += float(strs[-1])
    print(f'sum: {sum}, count: {cnt}, avg: {sum/cnt}')




