
import numpy as np
import pandas as pd 
import pickle
import os
import datetime
import math
import random
import warnings
import scipy.optimize
import sklearn.metrics


NEIGHBORS_PATH = 'GlassLabData/PT_TW_fits/Polished_Diffy' 
POLISHED_DIRECT_PATH = 'GlassLabData/PT_TW_fits/Polished_Direct'
DIRECT_SAVE_PATH = 'GlassLabData/PT_TW_fits/DIRECT_fits'
DIFEV_SAVE_PATH = 'GlassLabData/PT_TW_fits/Diffy_evolution_fits'
BASIN_HOPPING_SAVE_PATH = 'GlassLabData/PT_fits/BASIN_HOPPING_FITS'
SHGO_SAVE_PATH = 'GlassLabData/PT_fits/SHGO_FITS'
EXHUASTIVE_SAVE_PATH = 'GlassLabData/PT_TW_fits/EXHAUSTIVE_FITS'
BOBBY_PATH = 'GlassLabData/PT_TW_fits/BOBBY_FITS'
POLISH0_PATH = 'GlassLabData/PT_TW_fits/'
SCIPY_BRUTE_PATH = 'GlassLabData/PT_TW_fits/scipy_brute'

from datetime import date
def save_fit(fit, participant,algo):
  tdy=date.today()
  sp=""
  if algo=="direct":
    sp=DIRECT_SAVE_PATH
  elif algo=="diffev":
    sp=DIFEV_SAVE_PATH
  elif algo=="basin_hopping":
    sp=BASIN_HOPPING_SAVE_PATH
  elif algo=="shgo":
    sp=SHGO_SAVE_PATH
  elif algo=="brute":
    sp=EXHUASTIVE_SAVE_PATH
  elif algo=='bobby':
    sp=BOBBY_PATH
  elif algo=='neighbors':
    sp=NEIGHBORS_PATH
  elif algo=='polished_direct':
    sp=POLISHED_DIRECT_PATH
  elif algo=='scipy_brute_PTTW':
    sp=SCIPY_BRUTE_PATH



  fname=sp+"participant " + str(participant)+" "+tdy.strftime("%B %d, %Y")+".pickle"
  pickle_out=open(fname,"wb")
  pickle.dump(fit,pickle_out)
  pickle_out.close()

def load_fit(path):
  pickle_in=open(path,"rb") # put the path here
  fit=pickle.load(pickle_in)
  return fit

p = {
    1: 0.03,
    2: 0.06,
    3: 0.09,
    4: 0.12,
    5 : 0.14,
    6: 0.11,
    7: 0.09,
    8: 0.08,
    9: 0.07,
    10: 0.06,
    11: 0.05,
    12: 0.04,
    13: 0.03,
    14: 0.02,
    15: 0.01
}
I=15

def EUTv1(day, i_d, n, stored ):
    s=0
    for j in p.keys:
      s+=(j*(p[j]*10))

    util = n*i_d + (stored-n)*s
    return util

def cutoff_EUT(x):
    if x >= 32:
        return 14
    elif x in range(17, 32):
        return 13
    elif x in range(10, 17):
        return 12
    elif x in range(7, 10):
        return 11
    elif x in range(5, 7):
        return 10
    elif x == 4:
        return 9
    elif x == 3:
        return 8
    elif x == 2:
        return 7
    elif x == 1:
        return 1

all_data= pd.read_csv('All Data EUT 230519.csv')

SellData= all_data.copy()

columns=list(SellData.columns)
columns

for string in columns:
  if "Predicted"  in string:
      columns.remove(string)
print(columns)

error_cols=columns[-10:]

columns.remove('OEHold')
columns.remove('OHold')

columns.remove('OESold')
columns.remove('OSold')
columns.remove('MSold')

columns.remove('Diff')
columns.remove('Tot')
columns.remove('MDev')
columns.remove('MDev0')
columns.remove('Tot0')

SellData=all_data[columns]
SellData
error_data=all_data[error_cols]

df_days=[]
df_start=SellData.iloc[:,[0]]
splitting_df= SellData.drop("Subject", axis=1)

def split(df):
  cols=list(df.columns)
  stored_cols=[x for x in cols if ("Stored" in x)]

  for name in stored_cols:
    print("Stored+ day: ", name)

    stored_i=cols.index(name)
    print("stored index",stored_i)

    new_df= df.iloc[:, list(range(stored_i,stored_i+3))]
    print("length of new df : ", len(new_df.columns))
    print("columns of new df : ", list(new_df.columns))
    new_df= pd.concat([df_start,new_df], axis=1)
    df_days.insert(stored_i, new_df)
  print("length of df_days: ", len(df_days))

split(splitting_df)

df_days.reverse()

DAYS=list(range(0,68))

def prelec(p, gamma):
    x=math.exp(-(-math.log(p)) ** gamma)

    return x

prelec(p[4], 1.75)

def PTv2(day, price, sold, alpha, beta, lam, gam, c_o ):
  # this is our code and it is copied straight from what's shown on the report. PTv3 is a "working" version however it doesn't adhere directly to the model in the report
  gain=sum(  (prelec(  p[j]+ sum(h_probPTv2(day-1,f,c_o,gam) for f in range(day-2,1,-1)) * p[j], gam ) * (sold*(price-j))**alpha )   for j in range(c_o[day-1],price))

 # print("gain for day", day,"is ", gain)
  loss=sum((prelec(  p[j]+ sum(h_probPTv2(day-1,f,c_o,gam) for f in range(day-2,1,-1)) * p[j] , gam) * lam*(sold*(j-price))**beta )   for j in range(max(c_o[day-1],price+1),I+1))
  #print("loss for day", day,"is ", loss)
  util=gain-loss
 # print("util for day", day, "is", util)
  return util

def h_probPTv2(day, f, c_o,gam):
  prob=1.0
  for h in range(1,f+1):
    prob*=sum(prelec(p[j],gam) for j in range(1,c_o[day-h]))
  return prob

def obj_gain(day, price, sold, alpha, beta, lam, gam, c_o,j):
  gain=sold*(price-j)
  gain=gain**alpha
  gain*=prelec(p[j],gam)
  return gain

def obj_loss(day, price, sold, alpha, beta, lam, gam, c_o,j):
  loss=sold*(j-price)
  loss=loss**beta
  loss*=lam*prelec(p[j],gam)
  return loss

def PTv3(day,price,sold,alpha,beta,lam,gam,c_o):
  util=0
  for j in range(c_o[day-1],price):
    util+=obj_gain(day,price,sold,alpha,beta,lam,gam,c_o,j)


  for j in range(max(price+1,c_o[day-1]), I+1):

    util-=obj_loss(day,price,sold,alpha,beta,lam,gam,c_o,j)


  for k in range(1,day+2):

    h_prob=h_probPTv2(day,k-1,c_o,gam)
    #h_prob=h_probPT(day_f=k, actual_day=day,gamma=gam,cutoff=c_o)

    gain_prob=0
    for j in range(c_o[day-k],price):
      gain_prob+=obj_gain(day,price,sold,alpha,beta,lam,gam,c_o,j)

    loss_prob=0
    for j in range(max(price+1,c_o[day-k]), I+1):
      loss_prob+=obj_loss(day,price,sold,alpha,beta,lam,gam,c_o,j)

    util+=h_prob*(gain_prob-loss_prob)

  #print("util for day",day,"is ", util)
  return util

def max_units(day,price, stored,a,b,l,g,cut_offs):
  sells=[]
  pred=0
  greatest=-100

  for units in range(1,stored+1):

    prosp=PTv3(day,price,units,a,b,l,g,cut_offs)

    if prosp>greatest:
      greatest=prosp
      pred=units
      sells.append(units)

#  print("all possibile units to sell at", sells)

 # print("len of possible sells:", len(sells))

    #if prosp>greatest:
       # greatest=prosp
       # pred=units
  return pred

def info_return(day, participant):
  curr_df= df_days[day].iloc[participant] # data acquiring is fine.
  n_1=curr_df["Sold"+str(day+1)]
  #print("Sold units on day :",day, " is : ", n_1)

  N_d=curr_df["Stored"+str(day+1)]
  #print("Stored on day",day,"is:", N_d)
  i_d=curr_df["Price"+str(day+1)]
 # print("Price on day",day,"is",i_d)
  return(n_1,N_d,i_d)

def cutoff_pt_list(a,b,l,g):#this function generates a list before we make our predicitons
  cutoffs=[1]
  for day in DAYS:#0,1,2,3
    if day==0:
      continue
    price=cutoffs[day-1]
    prosp=PTv3(day,price,1,a,b,l,g,cutoffs)
    #prosp=PTv2(day,price,1,a,b,l,g,cutoffs)
    counts=0
    while (prosp <= 0 and price < 15):
                # Continually increment price until it is worthwhile to sell
                price += 1
                counts+=1
                prosp = PTv3(day,price,1,a,b,l,g,cutoffs)
                #prosp=PTv2(day,price,1,a,b,l,g,cutoffs)



    cutoffs.append(price)
   # print("counts from the previous value of", cutoffs[day-1], " is ", counts,"cutoff price is now", price, "for day ",day)
  return cutoffs

def fitPTv2(participant,a,b,l,g):
  predictions=[]                 #[0]*68
  #cut_offs=c_o
  cut_offs=cutoff_pt_list(a,b,l,g)            #cutoff_PT(a,b,l,g,participant)


  for day in DAYS: # 1,2,3
    pred=0
    vals=info_return(day, participant) # sold, stored,price
    stored=vals[1]
    price=vals[2]

    ''' c_o=cutoff_PT(a,b,l,g,day,cut_offs,stored)
    print("cut off price for day", day,"is ", c_o)
    cut_offs[day-1]=c_o'''
    if day==0:
      pred=stored
    else:

      if price>=cut_offs[day]:

        pred=max_units(day,price,stored,a,b,l,g, cut_offs)
      else:
        print(".")

    predictions.append(pred)
    #predictions.insert(day-1,pred)
  pred_series=pd.Series(predictions)
  #predictions_df['Participant '+str(participant+1)]=pred_series.values
 # predictions_df['Participant' +str(participant+1)+' cutoffs']=pd.Series(cut_offs).values
  print("DATA ADDED :)")
  return predictions,cut_offs

def PT_TW(day,price,sold,alpha,beta,lam,gam,tw,c_o):
  t=int(tw)

  if t>day: # 68,67,66
    util=PTv3(day,price,sold,alpha,beta,lam,gam,c_o)
    print("util PT : ", util)
  else:
    gain=sum(prelec(p[j] + sum((sum(p[k] for k in range(1,c_o[t-1])))**h for h in range(1,t-1)) * p[j], gam)* (sold*(price-j))**alpha for j in range(c_o[t-1],price))
    loss=sum(prelec(p[j] + sum((sum(p[k] for k in range(1,c_o[t-1])))**h for h in range(1,t-1)) * p[j],gam)*lam*(sold*(j-price))**beta for j in range(max(c_o[t-1],price+1),I+1))
    util=gain-loss
    print("util day:" ,day,util)
  return util

def cutoff_pt_tw_list(a,b,l,g,tw):
  t=tw
  cutoffs=[1]
  for day in DAYS:#0,1,2,3
    if day==0: # last day
      continue
    price=cutoffs[day-1]
    if day<t:
          prosp=PTv3(day,price,1,a,b,l,g,cutoffs)
    else:
          prosp = PT_TW(day,price,1,a,b,l,g,t,cutoffs)
    print(prosp)
    #prosp=PTv2(day,price,1,a,b,l,g,cutoffs)

    counts=0
    while (prosp <= 0 and price < I):
                # Continually increment price until it is worthwhile to sell
                price += 1
                counts+=1
                if day<t:
                  prosp=PTv3(day,price,1,a,b,l,g,cutoffs)
                else:
                  prosp = PT_TW(day,price,1,a,b,l,g,t,cutoffs)
                print("prosp on day", day, "is ", prosp)
                #prosp=PTv2(day,price,1,a,b,l,g,cutoffs)



    cutoffs.append(price)
    print("counts from the previous value of", cutoffs[day-1], " is ", counts,"cutoff price is now", price, "for day ",day)
  return cutoffs

def max_units_PTTW(day,price,stored,a,b,l,g,tw,cutoffs ):
  sells=[]
  pred=0
  greatest=-100

  for units in range(1,stored+1):

    prosp=PT_TW(day,price,units,a,b,l,g,tw,cutoffs)

    if prosp>greatest:
      greatest=prosp
      pred=units
      sells.append(units)

#  print("all possibile units to sell at", sells)

 # print("len of possible sells:", len(sells))

    #if prosp>greatest:
       # greatest=prosp
       # pred=units
  return pred

### Functions we are optimizing ###

def fitPT_TW(participant,a,b,l,g,tw):
  predictions=[]                 #[0]*68
  cut_offs=cutoff_pt_tw_list(a,b,l,g,tw)
  #cut_offs=cutoff_pt_tw_list(a,b,l,g,tw)            #cutoff_PT(a,b,l,g,participant)

  rev_d=reversed(DAYS)
  #rev_d=DAYS
  for day in rev_d: # 67,66,65...
    pred=0
    vals=info_return(day, participant) # sold, stored,price
    stored=vals[1]
    price=vals[2]
    if day==0:
      pred=stored
    else:

      if price>=cut_offs[day]:

        pred=max_units_PTTW(day,price,stored,a,b,l,g,tw,cut_offs)
      else:
        print("cutoff value of", cut_offs[day], "bigger than price", price, "HOLD PREDICTED ")

    predictions.append(pred)
    #predictions.insert(day-1,pred)
  pred_series=pd.Series(predictions)
  #predictions_df['Participant '+str(participant+1)]=pred_series.values
 # predictions_df['Participant' +str(participant+1)+' cutoffs']=pd.Series(cut_offs).values
  print("DATA ADDED :)")
  return predictions,cut_offs


soldOnly = SellData
for name in SellData.columns :
  if 'Sold' not in name:
    soldOnly=soldOnly.drop(name, axis=1)

soldOnly = soldOnly.T
soldOnly = soldOnly.iloc[::-1]
soldOnly

list(soldOnly[0].values)

from sklearn.metrics import mean_squared_error
def d2a_error(params,participant):
  a=params[0]
  b=params[1]
  l=params[2]
  g=params[3]


  vals=fitPTv2(participant,a,b,l,g)
  preds=vals[0]
  actuals=list(soldOnly[participant].values)
  mse=mean_squared_error(actuals,preds)
  error=d2a(actuals,preds)

  return 1-error


from sklearn.metrics import d2_absolute_error_score as d2a
def fit_errorPT_TW(params,participant):
  a=params[0]
  b=params[1]
  l=params[2]
  g=params[3]
  tw=params[4]



  vals=fitPT_TW(participant,a,b,l,g,tw)
  preds=vals[0]
  actuals=list(soldOnly[participant].values)
  actuals=actuals[::-1]

  error=d2a(actuals,preds)

  return 1-error

def fit_errorPTTW_dir(params,participant):
  a=params[0]
  b=params[1]
  l=params[2]
  g=params[3]
  tw=params[4]
  tw=int(tw)


  vals=fitPT_TW(participant,a,b,l,g,tw)
  preds=vals[0]
  actuals=list(soldOnly[participant].values)

  error=d2a(actuals,preds)

  return error

### main error func we are optimizing ### 

def count_errorPT_TW(params,participant):
  a=params[0]
  b=params[1]
  l=params[2]
  g=params[3]
  tw=params[4]
  tw=int(tw)

  vals=fitPT_TW(participant,a,b,l,g,tw)
  preds=vals[0]
  total=0
  d0=0
  for day in DAYS[::-1]:
    vals=info_return(day,participant)

    sold=vals[0]
    pred=preds[67-day]
    day_err=abs(sold-pred)
    total+=day_err
  return total

check=lambda params: params if params[0]<=params[1] else "Alpha greater than Beta"

def save_scipy_fit(fit):
   with open('/scratch/bfw20/pickle_saves/' + 'part_1_test_2.pkl', 'wb') as f:
      pickle.dump(fit, f)

from scipy.optimize import brute
def scipy_brute_PTTW(participant):

  part = participant
  rranges = (slice(0,1,0.1), slice(0,1,0.1), slice(1,2,0.1), slice(0,1,0.1), slice(2,68,1))

  result = brute(func=fit_errorPT_TW, ranges=rranges, args=(part,), Ns=10, workers=10)
  

  # idea here is to get the params of current fit and compare it to the previous fit and see if it's better
  # if it is we set current fit to best fit
  # if it isn't we keep the best fit as the best fit
  # then we return the best fit

  #create empty array for best fit  
  best_fit = result[0] #best fit
  save_scipy_fit(best_fit)

  # print (result.message)
 
  return best_fit

if __name__ == '__main__':
   bf1 = scipy_brute_PTTW(0)

