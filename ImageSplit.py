import cv2
import numpy as np

class imageSplit():

    def imageSplit(self, imageFilePath,outputPath,Size):
        array = self.iamgeConvertArray(imageFilePath=imageFilePath)
        setList = self.ArrayToSetList(array)
        i = 0
        for smallImage in setList:
            i+=1
            self.CreateImage(smallImage,outputPath+str(i)+".png",Size)

    '''
    create splited image to "./tmp/"
    '''
    def CreateImage(self,coordinates,fileName,Size):
        rowMin,rowMax,columnMin,columnMax = 0,0,0,0
        for coor in coordinates:
            rowMin,columnMin = coor
            rowMax,columnMax = coor
        for coor in coordinates:
            if coor[0]<rowMin : rowMin = coor[0]
            if coor[0]>rowMax : rowMax = coor[0]
            if coor[1]>columnMax : columnMax = coor[1]
            if coor[1]<columnMin : columnMin = coor[1]
        row = rowMax-rowMin
        colunm = columnMax-columnMin
        if row==0 or colunm==0:
            return
        img = np.zeros((row,colunm,3))
        img.fill(255)
        #print(img)
        for i in range(row):
            for j in range(colunm):
                if (i+rowMin,j+columnMin) in coordinates :
                    img[i,j,0] = 0
                    img[i,j,1] = 0
                    img[i,j,2] = 0
        img = cv2.resize(img,(Size,Size))
        cv2.imwrite(fileName,img)
        return



    def iamgeConvertArray(self, imageFilePath):
        img = cv2.imread(imageFilePath)
        #print(img.shape)
        #        cv2.imshow("image",img)
        #        cv2.waitKey(0)
        #        cv2.destroyAllWindows()
        row, column, t = img.shape
        array = [[0 for i in range(column)] for j in range(row)]
        count = 0
        for i in range(row):
            for j in range(column):
                r = img[i, j, 0]
                b = img[i, j, 1]
                g = img[i, j, 2]
                if r == 255 and b == 255 and g == 255:
                    count += 0
                    array[i][j] = 1
        #print(count)
        return array

    def ArrayToSetList(self, array):
        '''
        array is a 2D int array
        return a list[Set{(x,y)}],each set in list will split from the picture array.
        '''
        list = []
        row = len(array)
        column = len(array[0])
        mark = [[False for i in range(column)] for j in range(row)]
        for j in range(column):
            for i in range(row):
                s = set()
                stack = [(i,j)]
                while not len(stack)==0:
                    r,c = stack.pop()
                    if not mark[r][c] :
                        if array[r][c]==1:
                            s.add((r,c))
                            if r+1<row : stack.append((r+1,c))
                            if r-1>0 : stack.append((r-1,c))
                            if c+1<column : stack.append((r,c+1))
                            if c-1>0 : stack.append((r,c-1))
                        mark[r][c] = True
                if len(s)!=0:
                    list.append(s)
        #print(list)
        return list
