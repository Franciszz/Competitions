for i in range(112,120):
    for j in range(105,110):
        for m in range(95,105):
            for n in range(105,115):
                df_predict_ind = RevenueTran(df_model_macro, df_xgb_ind, market_info,
                                             [i/100,j/100,m/100,n/100], 0.5, 0.7, 0.35)
                df_submit = RevenueCom(df_predict_ind, pre_list)
                df_submit.to_csv(path%'/47_152/submit/submit_%d_%d_%d_%d.csv'%(i,j,m,n),
                    encoding='utf-8',header=False,index=False)
scores, biass = [],[]
file_list = os.listdir(path % '47_152/submit')
for filename in file_list:
    df_submit = pd.read_csv(path % '47_152/submit/'+filename, header=None).\
        rename(columns={0:'TICKER_SYMBOL',1:'predict'})
    df_pre = df_submit.assign(TICKER_SYMBOL=df_submit.TICKER_SYMBOL.\
                apply(lambda x:x[:6]))
    bias, score, num = cal_score(df_revenue, df_pre, df_market)
    scores.append(score), biass.append(bias)
    print('File:'+filename+'\tBias:'+str(np.round(bias,4))+
          '\tScore:'+str(np.round(score,4))+'\tNums:'+str(num)+'\n')
  
    
df_predict_ind = RevenueTran(df_model_macro, df_xgb_ind, market_info,
                             [1.23, 1.07, 1.0, 1.18], 0.5, 0.75, 0.25)
df_predict_indu = RevenueTran(df_model_macro, df_xgb_indu, market_info,
                             [1.23, 1.07, 1.0, 1.18], 0.5, 0.75, 0.25)
df_submit1 = RevenueCom(df_predict_ind, pre_list)
df_submit2 = RevenueCom(df_predict_indu, pre_list)
for i in range(40,60):
    df_submit3 = pd.DataFrame(dict(TICKER_SYMBOL=df_submit1.TICKER_SYMBOL,
                                   PREDICT = (df_submit1.PREDICT*i+
                                              df_submit2.PREDICT*(100-i))/100))
    df_submit3.to_csv(path%'/47_152/submit/submit_%d.csv'%(i),
        encoding='utf-8',header=False,index=False)
               
file_list = os.listdir(path % '47_152/submit')
print('\n')
### 遍历submit文件夹下的csv文件，计算score
for filename in file_list:
    df_submit = pd.read_csv(path % '47_152/submit/'+filename, header=None).\
        rename(columns={0:'TICKER_SYMBOL',1:'predict'})
    df_pre = df_submit.assign(TICKER_SYMBOL=df_submit.TICKER_SYMBOL.\
                apply(lambda x:x[:6]))
    bias, score, num = cal_score(df_revenue, df_pre, df_market)
    print('File:'+filename+'\tBias:'+str(np.round(bias,4))+
          '\tScore:'+str(np.round(score,4))+'\tNums:'+str(num)+'\n')
