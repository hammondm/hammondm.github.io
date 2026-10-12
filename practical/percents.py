
d = """
20
140
751
1389
"""

d2 = """
27
1728
110592
7077888
"""

def dovals(q):
	nums = [int(x) for x in q.split('\n') if len(x) > 0]
	t = sum(nums)
	res = [x/t for x in nums]
	for a,b in zip(nums,res):
		print(f"{a}: {b:.7f}")

dovals(d)
dovals(d2)

