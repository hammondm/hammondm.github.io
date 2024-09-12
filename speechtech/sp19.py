from sklearn import linear_model
import numpy as np

d = np.array([[-3,-1, 1],[-1, 1, 2],
              [ 1, 2, 3],[ 2, 3, 4],
              [ 3, 4, 3],[ 4, 3, 2],
              [ 3, 2, 0],[ 2, 0,-2]])

#create and fit model
r = linear_model.LinearRegression()
r.fit(d[:,:2],d[:,2])
#get intercept and coefficients
print(f'intercept:\t{r.intercept_:.2}')
print(f'coefficients:\t{r.coef_[0]:.2},' +
      f' {r.coef_[1]:.2}')
#get predicted values
yhat = r.predict(d[:,:2])
for n,y in enumerate(yhat):
	print(f'{d[n,0]:2},{d[n,1]:2}: y =' +
		    f' {d[n,2]:2}, y^ = {yhat[n]:4.2}')
