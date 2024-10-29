from PyQt6.QtCore import QCoreApplication, QSize, Qt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox, QGridLayout, QGroupBox
from PyQt6.QtGui import QCursor, QPixmap, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import matplotlib.pylab as plt
import os
import sys
import joblib
import numpy as np

df = None
result = None
confidenceLevels = None

if getattr(sys, 'frozen', False):
    # If the application is frozen (i.e., running as an executable)
    base_path = sys._MEIPASS

else:
    # If running as a script
    base_path = os.path.dirname(__file__)


class MainWindow(QWidget):
    # main
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Machine learning analysis for 5 parameters') # window name
        iconPath = os.path.join(base_path, 'pictures', 'wd_logo2.ico')
        icon = QIcon(iconPath)
        self.setWindowIcon(icon)
        # self.setFixedSize(QSize(1380, 780)) # window size

        # set layout
        layout = QGridLayout()
        layout.setObjectName('layout')
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        self.setLayout(layout)

        box = QGroupBox()
        box.setObjectName("stats-box")
        boxLayout = QGridLayout()
        box.setLayout(boxLayout)
        box.setFixedHeight(200)

        # create widgets
        headText = QLabel('BPI Bench Test csv log analyzer')
        headText.setObjectName('normal-text')
        fileName = QLabel('File Name:')
        fileName.setObjectName('normal-text')
        self.specificFileName = QLabel('')
        self.specificFileName.setObjectName('normal-text')
        browseBtn = QPushButton('Browse')
        browseBtn.setObjectName('btn')
        browseBtn.setFixedSize(73,28)
        browseBtn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        browseBtn.clicked.connect(self.browse_file)

        self.plot1_created = False
        self.plot2_created = False
        self.plot3_created = False
        self.plot4_created = False
        # self.plot5_created = False
        # self.plot6_created = False
        self.fig1, self.ax1 = plt.subplots(figsize=(8, 5))
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1_2 = self.ax1.twinx()
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5))
        self.canvas2 = FigureCanvas(self.fig2)
        self.fig3, self.ax3 = plt.subplots(figsize=(8, 5))
        self.canvas3 = FigureCanvas(self.fig3)
        self.fig4, self.ax4 = plt.subplots(figsize=(8, 5))
        self.canvas4 = FigureCanvas(self.fig4)
        self.ax4_2 = self.ax4.twinx()
        # self.fig5, self.ax5 = plt.subplots(figsize=(5, 2))
        # self.canvas5 = FigureCanvas(self.fig5)
        # self.fig6, self.ax6 = plt.subplots(figsize=(5, 2))
        # self.canvas6 = FigureCanvas(self.fig6)
        self.check_and_set_black_box()

        name = QLabel('')
        ampCell = QLabel('AmpCell(amp)')
        ampCell.setObjectName('bold-text')
        volts24 = QLabel('Volts24')
        volts24.setObjectName('bold-text')
        cellPower = QLabel('Cell Power')
        cellPower.setObjectName('bold-text')
        driveFanDmd = QLabel('DriveFan_dmd')
        driveFanDmd.setObjectName('bold-text')
        driveFanRpm = QLabel('DriveFan_rpm')
        driveFanRpm.setObjectName('bold-text')
        electronicFanDmd = QLabel('ElectronicFam_dmd')
        electronicFanDmd.setObjectName('bold-text')
        electronicFanRpm = QLabel('ElectronicFam_rpm')
        electronicFanRpm.setObjectName('bold-text')
        ftemp0 = QLabel('Ftemp0')
        ftemp0.setObjectName('bold-text')
        ftemp1 = QLabel('Ftemp1')
        ftemp1.setObjectName('bold-text')
        heatDmd = QLabel('Heat dmd')
        heatDmd.setObjectName('bold-text')

        max = QLabel('Max')
        max.setObjectName('bold-text')
        self.ampCellMax = QLabel('')
        self.ampCellMax.setObjectName('normal-text')
        self.volts24Max = QLabel('')
        self.volts24Max.setObjectName('normal-text')
        self.cellPowerMax = QLabel('')
        self.cellPowerMax.setObjectName('normal-text')
        self.driveFanDmdMax = QLabel('')
        self.driveFanDmdMax.setObjectName('normal-text')
        self.driveFanRpmMax = QLabel('')
        self.driveFanRpmMax.setObjectName('normal-text')
        self.electronicFanDmdMax = QLabel('')
        self.electronicFanDmdMax.setObjectName('normal-text')
        self.electronicFanRpmMax = QLabel('')
        self.electronicFanRpmMax.setObjectName('normal-text')
        self.ftemp0Max = QLabel('')
        self.ftemp0Max.setObjectName('normal-text')
        self.ftemp1Max = QLabel('')
        self.ftemp1Max.setObjectName('normal-text')
        self.heatDmdMax = QLabel('')
        self.heatDmdMax.setObjectName('normal-text')

        mean = QLabel('Mean')
        mean.setObjectName('bold-text')
        self.ampCellMean = QLabel('')
        self.ampCellMean.setObjectName('normal-text')
        self.volts24Mean = QLabel('')
        self.volts24Mean.setObjectName('normal-text')
        self.cellPowerMean = QLabel('')
        self.cellPowerMean.setObjectName('normal-text')
        self.driveFanDmdMean = QLabel('')
        self.driveFanDmdMean.setObjectName('normal-text')
        self.driveFanRpmMean = QLabel('')
        self.driveFanRpmMean.setObjectName('normal-text')
        self.electronicFanDmdMean = QLabel('')
        self.electronicFanDmdMean.setObjectName('normal-text')
        self.electronicFanRpmMean = QLabel('')
        self.electronicFanRpmMean.setObjectName('normal-text')
        self.ftemp0Mean = QLabel('')
        self.ftemp0Mean.setObjectName('normal-text')
        self.ftemp1Mean = QLabel('')
        self.ftemp1Mean.setObjectName('normal-text')
        self.heatDmdMean = QLabel('')
        self.heatDmdMean.setObjectName('normal-text')

        min = QLabel('Min')
        min.setObjectName('bold-text')
        self.ampCellMin = QLabel('')
        self.ampCellMin.setObjectName('normal-text')
        self.volts24Min = QLabel('')
        self.volts24Min.setObjectName('normal-text')
        self.cellPowerMin = QLabel('')
        self.cellPowerMin.setObjectName('normal-text')
        self.driveFanDmdMin = QLabel('')
        self.driveFanDmdMin.setObjectName('normal-text')
        self.driveFanRpmMin = QLabel('')
        self.driveFanRpmMin.setObjectName('normal-text')
        self.electronicFanDmdMin = QLabel('')
        self.electronicFanDmdMin.setObjectName('normal-text')
        self.electronicFanRpmMin = QLabel('')
        self.electronicFanRpmMin.setObjectName('normal-text')
        self.ftemp0Min = QLabel('')
        self.ftemp0Min.setObjectName('normal-text')
        self.ftemp1Min = QLabel('')
        self.ftemp1Min.setObjectName('normal-text')
        self.heatDmdMin = QLabel('')
        self.heatDmdMin.setObjectName('normal-text')

        std = QLabel('Std')
        std.setObjectName('bold-text')
        self.ampCellStd = QLabel('')
        self.ampCellStd.setObjectName('normal-text')
        self.volts24Std = QLabel('')
        self.volts24Std.setObjectName('normal-text')
        self.cellPowerStd = QLabel('')
        self.cellPowerStd.setObjectName('normal-text')
        self.driveFanDmdStd = QLabel('')
        self.driveFanDmdStd.setObjectName('normal-text')
        self.driveFanRpmStd = QLabel('')
        self.driveFanRpmStd.setObjectName('normal-text')
        self.electronicFanDmdStd = QLabel('')
        self.electronicFanDmdStd.setObjectName('normal-text')
        self.electronicFanRpmStd = QLabel('')
        self.electronicFanRpmStd.setObjectName('normal-text')
        self.ftemp0Std = QLabel('')
        self.ftemp0Std.setObjectName('normal-text')
        self.ftemp1Std = QLabel('')
        self.ftemp1Std.setObjectName('normal-text')
        self.heatDmdStd = QLabel('')
        self.heatDmdStd.setObjectName('normal-text')

        predictBtn = QPushButton('Predict')
        predictBtn.setObjectName('btn')
        predictBtn.setFixedSize(86,28)
        predictBtn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        predictBtn.clicked.connect(self.predict)

        footer = QLabel('csv log analyzer BPI v11')
        footer.setObjectName('footer-text')        

        # add widgets to the layout #
        # header
        layout.addWidget(headText, 0,0)
        layout.addWidget(browseBtn, 0,2, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(fileName, 0,5, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.specificFileName, 0,6, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # graph
        layout.addWidget(self.canvas1, 2, 0, 1, 5)
        layout.addWidget(self.canvas2, 2, 5, 1, 6)
        layout.addWidget(self.canvas3, 3, 0, 1, 5)
        layout.addWidget(self.canvas4, 3, 5, 1, 6)
        # layout.addWidget(self.canvas5, 3, 3, 1, 4)
        # layout.addWidget(self.canvas6, 3, 7, 1, 4)

        # table
        boxLayout.addWidget(name, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(ampCell, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(volts24, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(cellPower, 0, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(driveFanDmd, 0, 4, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(driveFanRpm, 0, 5, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(electronicFanDmd, 0, 6, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(electronicFanRpm, 0, 7, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(ftemp0, 0, 8, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(ftemp1, 0, 9, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(heatDmd, 0, 10, alignment=Qt.AlignmentFlag.AlignCenter)

        boxLayout.addWidget(max, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ampCellMax, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.volts24Max, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.cellPowerMax, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanDmdMax, 1, 4, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanRpmMax, 1, 5, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanDmdMax, 1, 6, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanRpmMax, 1, 7, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp0Max, 1, 8, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp1Max, 1, 9, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.heatDmdMax, 1, 10, alignment=Qt.AlignmentFlag.AlignCenter)

        boxLayout.addWidget(mean, 2, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ampCellMean, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.volts24Mean, 2, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.cellPowerMean, 2, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanDmdMean, 2, 4, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanRpmMean, 2, 5, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanDmdMean, 2, 6, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanRpmMean, 2, 7, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp0Mean, 2, 8, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp1Mean, 2, 9, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.heatDmdMean, 2, 10, alignment=Qt.AlignmentFlag.AlignCenter)

        boxLayout.addWidget(min, 3, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ampCellMin, 3, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.volts24Min, 3, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.cellPowerMin, 3, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanDmdMin, 3, 4, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanRpmMin, 3, 5, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanDmdMin, 3, 6, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanRpmMin, 3, 7, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp0Min, 3, 8, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp1Min, 3, 9, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.heatDmdMin, 3, 10, alignment=Qt.AlignmentFlag.AlignCenter)

        boxLayout.addWidget(std, 4, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ampCellStd, 4, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.volts24Std, 4, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.cellPowerStd, 4, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanDmdStd, 4, 4, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.driveFanRpmStd, 4, 5, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanDmdStd, 4, 6, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.electronicFanRpmStd, 4, 7, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp0Std, 4, 8, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.ftemp1Std, 4, 9, alignment=Qt.AlignmentFlag.AlignCenter)
        boxLayout.addWidget(self.heatDmdStd, 4, 10, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(box, 5, 0, 5, 11)  # Spans rows 5-9 and columns 0-10

        # predict
        layout.addWidget(predictBtn, 10, 10, alignment=Qt.AlignmentFlag.AlignLeft)
        self.prediction = None
        
        # footer
        layout.addWidget(footer, 11, 0 , alignment=Qt.AlignmentFlag.AlignLeft)

    # check plot
    def check_and_set_black_box(self):
        # Check if any plot was created and set the black box if not
        if not self.plot1_created:
            self.ax1.set_visible(False)
            self.ax1_2.set_visible(False)
        if not self.plot2_created:
            self.ax2.set_visible(False)
        if not self.plot3_created:
            self.ax3.set_visible(False)
        if not self.plot4_created:
            self.ax4.set_visible(False)
            self.ax4_2.set_visible(False)
        # if not self.plot5_created:
        #     self.ax5.set_visible(False)
        # if not self.plot6_created:
        #     self.ax6.set_visible(False)

        # Draw the canvases to reflect the changes
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()
    # browse file
    def browse_file(self):
        global df
        filePath, _ = QFileDialog.getOpenFileName(self, "Select File")
        csvName = os.path.basename(filePath)
        self.specificFileName.setText(csvName)
        
        if filePath:
            self.filePath = filePath
            df = pd.read_csv(self.filePath)
            df1 = df[df['FlowTemp_Tenths'] >= 35]

            # Clear previous plots
            self.ax1.clear()
            self.ax1_2.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            self.ax4_2.clear()

            self.ax1.plot(df1['HeatDmd'].index, df1['HeatDmd'], color='orange', label='HeatDmd')
            self.ax1_2.plot(df1['CellTotalHeater_mA'].index, df1['CellTotalHeater_mA'], color='green', label='CellHeatTotal-mA')
            # self.ax1.set_ylabel('celsius')
            # self.ax1_2.set_ylabel('mAmp')
            self.ax1.set_xlabel('time (min)')
            self.ax1.legend(loc='upper left')
            self.ax1_2.legend(loc='upper right')
            self.ax1.set_visible(True)
            self.ax1_2.set_visible(True)
            self.ax1.grid(linestyle='--')
            self.canvas1.draw()

            self.ax2.plot(df1['FTemp'].index, df1['FTemp'], color='orange', label='FTemp')
            self.ax2.plot(df1['FlowTemp_Tenths'].index, df1['FlowTemp_Tenths'], color='green', label='FlowTemp_Tenths')
            # self.ax2.set_ylabel('celsius')
            self.ax2.set_xlabel('time (min)')
            self.ax2.legend(loc='upper left')
            self.ax2.set_visible(True)
            self.ax2.grid(linestyle='--')
            self.canvas2.draw()

            self.ax3.plot(df1['ElectronicsFan_MeasuredRPM_2'].index, df1['ElectronicsFan_MeasuredRPM_2'], color='orange', label='ElectronicsFan_MeasuredRPM_2')
            self.ax3.plot(df1['DriveFan_MeasuredRPM'].index, df1['DriveFan_MeasuredRPM'], color='green', label='DriveFan_MeasuredRPM')
            # self.ax3.set_ylabel('RPM')
            self.ax3.set_xlabel('time (min)')
            self.ax3.legend(loc='center')
            self.ax3.set_visible(True)
            self.ax3.grid(linestyle='--')
            self.canvas3.draw()

            self.ax4.plot(df1['HeatDmd'].index, df1['HeatDmd'], color='orange', label='HeatDmd')
            self.ax4.plot(df1['CoolDmd'].index, df1['CoolDmd'], color='blue', label='CoolDmd')
            self.ax4_2.plot(df1['FlowTemp_Tenths'].index, df1['FlowTemp_Tenths'], color='green', label='FlowTemp_Tenths')
            # self.ax4.set_ylabel('PWM')
            # self.ax4_2.set_ylabel('celsius')
            self.ax4.set_xlabel('time (min)')
            self.ax4.legend(loc='upper left')
            self.ax4_2.legend(loc='upper right')
            self.ax4.set_visible(True)
            self.ax4_2.set_visible(True)
            self.ax4.grid(linestyle='--')
            self.canvas4.draw()
            
            self.ampCellMax.setText('%.2f'%((df1['CellTotalHeater_mA'].max())))
            self.volts24Max.setText('%.2f'%(df1['Volts24v'].max()))
            self.cellPowerMax.setText('%.2f'%(df1['CellTotalHeater_mA'].max() * df1['Volts24v'].max()))
            self.driveFanDmdMax.setText('%.2f'%(df1['DriveFan_Dmd'].max()))
            self.driveFanRpmMax.setText('%.2f'%(df1['DriveFan_MeasuredRPM'].max()))
            self.electronicFanDmdMax.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_1'].max()))
            self.electronicFanRpmMax.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_2'].max()))
            self.ftemp0Max.setText('%.2f'%(df1['FTemp'].max()))
            self.ftemp1Max.setText('%.2f'%(df1['FlowTemp_Tenths'].max()))
            self.heatDmdMax.setText('%.2f'%(df1['HeatDmd'].max()))
            
            self.ampCellMean.setText('%.2f'%((df1['CellTotalHeater_mA'].mean())))
            self.volts24Mean.setText('%.2f'%(df1['Volts24v'].mean()))
            self.cellPowerMean.setText('%.2f'%(df1['CellTotalHeater_mA'].mean() * df1['Volts24v'].mean()))
            self.driveFanDmdMean.setText('%.2f'%(df1['DriveFan_Dmd'].mean()))
            self.driveFanRpmMean.setText('%.2f'%(df1['DriveFan_MeasuredRPM'].mean()))
            self.electronicFanDmdMean.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_1'].mean()))
            self.electronicFanRpmMean.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_2'].mean()))
            self.ftemp0Mean.setText('%.2f'%(df1['FTemp'].mean()))
            self.ftemp1Mean.setText('%.2f'%(df1['FlowTemp_Tenths'].mean()))
            self.heatDmdMean.setText('%.2f'%(df1['HeatDmd'].mean()))
            
            self.ampCellMin.setText('%.2f'%((df1['CellTotalHeater_mA'].min())))
            self.volts24Min.setText('%.2f'%(df1['Volts24v'].min()))
            self.cellPowerMin.setText('%.2f'%(df1['CellTotalHeater_mA'].min() * df1['Volts24v'].min()))
            self.driveFanDmdMin.setText('%.2f'%(df1['DriveFan_Dmd'].min()))
            self.driveFanRpmMin.setText('%.2f'%(df1['DriveFan_MeasuredRPM'].min()))
            self.electronicFanDmdMin.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_1'].min()))
            self.electronicFanRpmMin.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_2'].min()))
            self.ftemp0Min.setText('%.2f'%(df1['FTemp'].min()))
            self.ftemp1Min.setText('%.2f'%(df1['FlowTemp_Tenths'].min()))
            self.heatDmdMin.setText('%.2f'%(df1['HeatDmd'].min()))
            
            self.ampCellStd.setText('%.2f'%((df1['CellTotalHeater_mA'].std())))
            self.volts24Std.setText('%.2f'%(df1['Volts24v'].std()))
            self.cellPowerStd.setText('%.2f'%(df1['CellTotalHeater_mA'].std() * df1['Volts24v'].std()))
            self.driveFanDmdStd.setText('%.2f'%(df1['DriveFan_Dmd'].std()))
            self.driveFanRpmStd.setText('%.2f'%(df1['DriveFan_MeasuredRPM'].std()))
            self.electronicFanDmdStd.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_1'].std()))
            self.electronicFanRpmStd.setText('%.2f'%(df1['ElectronicsFan_MeasuredRPM_2'].std()))
            self.ftemp0Std.setText('%.2f'%(df1['FTemp'].std()))
            self.ftemp1Std.setText('%.2f'%(df1['FlowTemp_Tenths'].std()))
            self.heatDmdStd.setText('%.2f'%(df1['HeatDmd'].std()))

    # prediction
    def predict(self):
        global result, confidenceLevels
        if df is None:
            QMessageBox.critical(self, 'Notification', 'You must browse the data')
        else:
            testDf = df.iloc[:30]
            testDf = testDf[testDf['FlowTemp_Tenths'] >= 35]
            print(testDf)
            scaler_path = os.path.join(base_path, 'models', 'scaler_5_parameter.pkl')
            model_path = os.path.join(base_path, 'models', 'knn_model.pkl')

            sc = joblib.load(scaler_path) #change path if it not run
            model = joblib.load(model_path) #change path if it not run
            HeatDmd = [np.mean(testDf['HeatDmd'])]
            TargetTemp_Tenths = [np.mean(testDf['TargetTemp_Tenths'])]
            FlowTemp_Tenths = [np.mean(testDf['FlowTemp_Tenths'])]
            CoolDmd = [np.mean(testDf['CoolDmd'])]
            FTemp = [np.mean(testDf['FTemp'])]
            dfDic = {'HeatDmd': HeatDmd, 'TargetTemp_Tenths': TargetTemp_Tenths, 'FlowTemp_Tenths': FlowTemp_Tenths, 
                    'CoolDmd': CoolDmd, 'FTemp': FTemp}
            test = pd.DataFrame(dfDic)
            dfTestSC = sc.transform(test)
            predictionTest = model.predict(dfTestSC)
            print(predictionTest)
            result = 'Fail' if predictionTest[0] == 0 else 'Pass'
            confidenceLevels = model.predict_proba(dfTestSC)
            # QMessageBox.information(self, 'Result', f'prediction {result}, confidence levels {confidenceLevels.max(axis=1)[0]}')
            self.prediction = predictWindow()
            self.prediction.show()

class predictWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Prediction')
        self.setFixedSize(QSize(500,200))
        layoutPredict = QGridLayout()
        layoutPredict.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setLayout(layoutPredict)

        topic = QLabel('AI prediction')
        topic.setObjectName('bold-text')

        resultText = QLabel('Result:')
        resultText.setObjectName('normal-text')

        img_path = os.path.join(base_path, 'pictures', 'pass.png') if result == 'Pass' else os.path.join(base_path, 'pictures', 'fail.png')
        img = QPixmap(img_path)#change path if it not run
        resizedImg = img.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio)
        labelImg = QLabel()
        labelImg.setPixmap(resizedImg)
        labelImg.setAlignment(Qt.AlignmentFlag.AlignCenter)

        confidenceText = QLabel('Confidence Level:')
        confidenceText.setObjectName('normal-text')

        maxConfidence = round(confidenceLevels.max(axis=1)[0] * 100, 2)
        confidence = QLabel(f'{maxConfidence} %')
        confidence.setObjectName('bold-text')

        if maxConfidence >= 80:
            confidence.setObjectName('confidence-high')
        else :
            confidence.setObjectName('confidence-low')

        closeBtn = QPushButton('Close')
        closeBtn.setFixedSize(74, 25)
        closeBtn.setObjectName('btn')
        closeBtn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        closeBtn.clicked.connect(self.closeWindow)
        
        layoutPredict.addWidget(topic, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        layoutPredict.addWidget(resultText, 2, 0, alignment=Qt.AlignmentFlag.AlignRight)
        layoutPredict.addWidget(labelImg, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        layoutPredict.addWidget(confidenceText, 2, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        layoutPredict.addWidget(confidence, 2, 3, alignment=Qt.AlignmentFlag.AlignLeft)
        layoutPredict.addWidget(closeBtn, 4, 3, alignment=Qt.AlignmentFlag.AlignRight)

    def closeWindow(self):
        self.close()

app = QCoreApplication.instance()
if app is None:
    app = QApplication([])
    stylesheet_path = os.path.join(base_path, 'styles', 'style.qss')
    with open(stylesheet_path, 'r') as style:
        app.setStyleSheet(style.read())

window = MainWindow()
window.show()
app.exec()