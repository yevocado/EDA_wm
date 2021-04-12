# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:28:59 2021

@author: user
"""
april = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/April/Product_Korean/Product_Korean2.csv')
May = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/May/Product_Korean/Product_Korean2.csv')
June = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/June/Product_Korean/Product_Korean2.csv')
July = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/July/Product_Korean/Product_Korean2.csv')
August = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/August/Product_Korean/Product_Korean2.csv')

August.isnull().sum() #칼럼별 결측치 구하기
August.count() #데이터개수세기

april['price']=april.quantity*april.unitPrice
May['price']=May.quantity*May.unitPrice
June['price']=June.quantity*June.unitPrice
July['price']=July.quantity*July.unitPrice
August['price']=August.quantity*August.unitPrice

August.describe() #데이터 describe
pd.options.display.float_format = '{:.5f}'.format
pd.reset_option('display.float_format') #지수표현(과학적표기법)초기화

month=[[4, april['price'].count(),april['price'].sum(), april1['price'].count()],
       [5, May['price'].count(),May['price'].sum(), May1['price'].count()],
       [6, June['price'].count(),June['price'].sum(),June1['price'].count()],
       [7, July['price'].count(),July['price'].sum(),July1['price'].count()],
       [8, August['price'].count(),August['price'].sum(),August1['price'].count()]]
month=pd.DataFrame(month, columns=['month','count_p','price','count_p_wo_0'])

l=np.arange(len(month))

plt.bar(month.month-0.15, month.count_p, width=0.3)
plt.bar(month.month+0.15, month.count_p_wo_0, width=0.3)
plt.xlabel('월', size=10)
plt.ylabel('구매횟수', size=10)
plt.legend()

plt.bar(l+0.3, centers2.Price, width=0.3)
plt.bar(month.month, month.count_p, width=0.5)

plt.bar(month.month, month.price, width=0.5, color='coral')
plt.xlabel('월', size=10)
plt.ylabel('판매가격', size=10)

plt.boxplot(april.price)

price=[april['price'],May['price'],June['price']]

april_0=april.loc[april['price']==0,:] #price가 0인 데이터
april1=april.drop(april_0.index)
april_0=april_0.groupby('lcate').size(); april_0=april_0.reset_index(level=['lcate'], inplace=False)
april_0.columns=['lcate','counts']; april_0=april_0.sort_values(by=['counts'],ascending=False)

May_0=May.loc[May['price']==0,:]#price가 0인 데이터
May1=May.drop(May_0.index) 
May_0=May_0.groupby('lcate').size(); May_0=May_0.reset_index(level=['lcate'], inplace=False)
May_0.columns=['lcate','counts']; May_0=May_0.sort_values(by=['counts'],ascending=False)

June_0=June.loc[June['price']==0,:] #price가 0인 데이터
June1=June.drop(June_0.index) 
June_0=June_0.groupby('lcate').size(); June_0=June_0.reset_index(level=['lcate'], inplace=False)
June_0.columns=['lcate','counts']; June_0=June_0.sort_values(by=['counts'],ascending=False)

July_0=July.loc[July['price']==0,:] #price가 0인 데이터
July1=July.drop(July_0.index)
July_0=July_0.groupby('lcate').size(); July_0=July_0.reset_index(level=['lcate'], inplace=False)
July_0.columns=['lcate','counts']; July_0=July_0.sort_values(by=['counts'],ascending=False)

August_0=August.loc[August['price']==0,:] #price가 0인 데이터
August1=August.drop(August_0.index) 
August_0=August_0.groupby('lcate').size(); August_0=August_0.reset_index(level=['lcate'], inplace=False)
August_0.columns=['lcate','counts']; August_0=August_0.sort_values(by=['counts'],ascending=False)

