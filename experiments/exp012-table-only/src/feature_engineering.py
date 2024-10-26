def feature_engineering(df):
    season_cols = [col for col in df.columns if 'Season' in col]
    df = df.drop(season_cols, axis=1) 
    df['BMI_Age'] = df['Physical-BMI'] * df['Basic_Demos-Age']
    df['Internet_Hours_Age'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['Basic_Demos-Age']
    df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
    df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
    df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
    df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
    
    return df

class Feature:
    def __init__(self):
        self.featuresCols = [
            'Basic_Demos-Age', 
            'Basic_Demos-Sex',
            'CGAS-CGAS_Score', 
            'Physical-BMI',
            'Physical-Height', 
            'Physical-Weight', 
            'Physical-Waist_Circumference',
            'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
            'Fitness_Endurance-Max_Stage',
            'Fitness_Endurance-Time_Mins', 
            'Fitness_Endurance-Time_Sec',
            'FGC-FGC_CU', 
            'FGC-FGC_CU_Zone', 
            'FGC-FGC_GSND',
            'FGC-FGC_GSND_Zone', 
            'FGC-FGC_GSD', 
            'FGC-FGC_GSD_Zone', 
            'FGC-FGC_PU',
            'FGC-FGC_PU_Zone',
            'FGC-FGC_SRL',
            'FGC-FGC_SRL_Zone',
            'FGC-FGC_SRR',
            'FGC-FGC_SRR_Zone',
            'FGC-FGC_TL',
            'FGC-FGC_TL_Zone',
            'BIA-BIA_Activity_Level_num',
            'BIA-BIA_BMC', 
            'BIA-BIA_BMI',
            'BIA-BIA_BMR', 
            'BIA-BIA_DEE', 
            'BIA-BIA_ECW', 
            'BIA-BIA_FFM',
            'BIA-BIA_FFMI', 
            'BIA-BIA_FMI', 
            'BIA-BIA_Fat', 
            'BIA-BIA_Frame_num',
            'BIA-BIA_ICW', 
            'BIA-BIA_LDM', 
            'BIA-BIA_LST', 
            'BIA-BIA_SMM',
            'BIA-BIA_TBW', 
            'PAQ_A-PAQ_A_Total',
            'PAQ_C-PAQ_C_Total', 
            'SDS-SDS_Total_Raw',
            'SDS-SDS_Total_T',
            'PreInt_EduHx-computerinternet_hoursday', 
            'sii', 
            'BMI_Age',
            'Internet_Hours_Age',
            'BMI_Internet_Hours',
            'BFP_BMI', 
            'FFMI_BFP', 
            'FMI_BFP', 
            'LST_TBW', 
            'BFP_BMR', 
            'BFP_DEE', 
            'BMR_Weight', 
            'DEE_Weight',
            'SMM_Height', 
            'Muscle_to_Fat', 
            'Hydration_Status', 
            'ICW_TBW',
        ]