# iterate.py

# @all:
D = {'Mary':231}
D['Bob'] = 123
D['Alice'] = 331
D['Ted'] = 987

for i in sorted(D):
    if i == 'Alice':
        continue
    if i == 'John':
        print("Loop ends. Cleese alert!")
        break;
    print(i+" "+str(D[i]))
else:
    print("Cleese is not in the list.")
# @:all
