class class_regression:
    '''Contains all the regression logic'''

    def get_regressor(self):
        print('Running regressors for the following datasets: \n')
        self.WineQuality()
        self.Communities_Crime()
        self.QSAR_aquatic_toxicity()
        self.Parkinson_Speech()
        self.Facebook_metrics()
        self.Bike_Sharing()
        self.Student_Performance()
        self.Concrete_Compressive_Strength()
        self.SGEMM_GPU_kernel_performance()
        self.Merck_Molecular_Activity_Challenge()