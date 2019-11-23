pp.pprint(stats)

print('mean same')
for sim in sim_st:
    mx = float(stats[sim]['vgg']['u'])
    mi = float(stats[sim]['vgg']['u'])
    for md in models:
        new_num = float(stats[sim][md]['u'])
        if(new_num > mx):
            mx = new_num
        if(new_num < mi):
            mi = new_num
    print(sim,mi,mx)

print('mean diff')
for sim in diff_st:
    mx = float(stats[sim]['vgg']['u'])
    mi = float(stats[sim]['vgg']['u'])
    for md in models:
        new_num = float(stats[sim][md]['u'])
        if(new_num > mx):
            mx = new_num
        if(new_num < mi):
            mi = new_num
    print(sim,mi,mx)
    
print('sd same')
for sim in sim_st:
    mx = float(stats[sim]['vgg']['s'])
    mi = float(stats[sim]['vgg']['s'])
    for md in models:
        new_num = float(stats[sim][md]['s'])
        if(new_num > mx):
            mx = new_num
        if(new_num < mi):
            mi = new_num
    print(sim,mi,mx)

print('sd diff')
for sim in diff_st:
    mx = float(stats[sim]['vgg']['s'])
    mi = float(stats[sim]['vgg']['s'])
    for md in models:
        new_num = float(stats[sim][md]['s'])
        if(new_num > mx):
            mx = new_num
        if(new_num < mi):
            mi = new_num
    print(sim,mi,mx)