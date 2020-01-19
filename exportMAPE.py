# import xlsxwriter module 
import xlsxwriter 
import numpy as np

def exportExcel(predict_mon, predict_sun, predict_rests, MAPE_mon, MAPE_sun, MAPE_rests, filename):
    """
    This functions export result to excel with Format required
    
    Parameters:
        predict (array): data predict result
        MAPE_mon (array): MAPE Monday result
        MAPE_sun (array): MAPE Sunday result
        MAPE_rests (array): MAPE Restsday result
        filename (string): filename excel to export
    """
    # create NN model    
    #
    workbook = xlsxwriter.Workbook(filename)
    # By default worksheet names in the spreadsheet will be  
    # Sheet1, Sheet2 etc., but we can also specify a name. 
    worksheet_predict = workbook.add_worksheet("Predict Data") 

    header = ['Thu','Ngay','0:00','1:00','2:00','3:00','4:00','5:00','6:00','7:00','8:00','9:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']

    #Write header
    for i in range(0, len(header)): 
      worksheet_predict.write(0, i, header[i]) 
    
    #Write data
    lenDates = predict_mon.shape[0] + predict_sun.shape[0] + predict_rests.shape[0]
    irests = 0
    isun = 0
    imon = 0
    #print(lenDates)
    for idate in range(lenDates):
      if (idate % 7 == 0):
        worksheet_predict.write(idate+1, 0, 2)
        for idata in range(24):
          worksheet_predict.write(idate+1, idata+2, predict_rests[irests,idata])
        irests +=1
        continue
      if (idate % 7 == 1):
        worksheet_predict.write(idate+1, 0, 3)
        for idata in range(24):
          worksheet_predict.write(idate+1, idata+2, predict_rests[irests,idata])
        irests +=1
        continue
      if (idate % 7 == 2):
        worksheet_predict.write(idate+1, 0, 4)
        for idata in range(24):
          worksheet_predict.write(idate+1, idata+2, predict_rests[irests,idata])
        irests +=1
        continue
      if (idate % 7 == 3):
        worksheet_predict.write(idate+1, 0, 5)
        for idata in range(24):
          worksheet_predict.write(idate+1, idata+2, predict_rests[irests,idata])
        irests +=1
        continue
      if (idate % 7 == 4):
        worksheet_predict.write(idate+1, 0, 6)
        for idata in range(24):
          worksheet_predict.write(idate+1, idata+2, predict_rests[irests,idata])
        irests +=1
        continue
      if (idate % 7 == 5):
        if isun<predict_sun.shape[0]:
          worksheet_predict.write(idate+1, 0, 7)
          for idata in range(24):
            worksheet_predict.write(idate+1, idata+2, predict_sun[isun,idata])
          isun +=1
        else:
            break
        continue
      if (idate % 7 == 6):
        if imon<predict_mon.shape[0]:
          worksheet_predict.write(idate+1, 0, 'CN')
          for idata in range(24):
            worksheet_predict.write(idate+1, idata+2, predict_mon[imon,idata])
          imon +=1
        else:
            break
        continue

    # Sheet1, Sheet2 etc., but we can also specify a name. 
    worksheet_MAPE = workbook.add_worksheet("MAPE") 

    header = ['Thu','Ngay','0:00','1:00','2:00','3:00','4:00','5:00','6:00','7:00','8:00','9:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00','MAPE_avgDay']

    #Write header
    for i in range(0, len(header)): 
      worksheet_MAPE.write(0, i, header[i]) 

    #Write data
    lenDates = MAPE_mon.shape[0] + MAPE_sun.shape[0] + MAPE_rests.shape[0]
    irests = 0
    isun = 0
    imon = 0
    #print(lenDates)
    for idate in range(lenDates):
      if (idate % 7 == 0):
        worksheet_MAPE.write(idate+1, 0, 2)
        for idata in range(24):
          worksheet_MAPE.write(idate+1, idata+2, MAPE_rests[irests,idata])
        worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_rests[irests,:])/24)
        irests +=1
        continue
      if (idate % 7 == 1):
        worksheet_MAPE.write(idate+1, 0, 3)
        for idata in range(24):
          worksheet_MAPE.write(idate+1, idata+2, MAPE_rests[irests,idata])
        worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_rests[irests,:])/24)
        irests +=1
        continue
      if (idate % 7 == 2):
        worksheet_MAPE.write(idate+1, 0, 4)
        for idata in range(24):
          worksheet_MAPE.write(idate+1, idata+2, MAPE_rests[irests,idata])
        worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_rests[irests,:])/24)
        irests +=1
        continue
      if (idate % 7 == 3):
        worksheet_MAPE.write(idate+1, 0, 5)
        for idata in range(24):
          worksheet_MAPE.write(idate+1, idata+2, MAPE_rests[irests,idata])
        worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_rests[irests,:])/24)
        irests +=1
        continue
      if (idate % 7 == 4):
        worksheet_MAPE.write(idate+1, 0, 6)
        for idata in range(24):
          worksheet_MAPE.write(idate+1, idata+2, MAPE_rests[irests,idata])
        worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_rests[irests,:])/24)
        irests +=1
        continue
      if (idate % 7 == 5):
        if isun<MAPE_sun.shape[0]:
          worksheet_MAPE.write(idate+1, 0, 7)
          for idata in range(24):
            worksheet_MAPE.write(idate+1, idata+2, MAPE_sun[isun,idata])
          worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_sun[isun,:])/24)
          isun +=1
        else:
            break
        continue
      if (idate % 7 == 6):
        if imon<MAPE_mon.shape[0]:
          worksheet_MAPE.write(idate+1, 0, 'CN')
          for idata in range(24):
            worksheet_MAPE.write(idate+1, idata+2, MAPE_mon[imon,idata])
          worksheet_MAPE.write(idate+1, 26, np.sum(MAPE_mon[imon,:])/24)
          imon +=1
        else:
            break
        continue
    
    workbook.close()
