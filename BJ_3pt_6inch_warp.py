class pt3_cal:
    def __init__(self,file_path ):

        self.file_path = file_path
        self.Bow_3pt = 0 
        self.Warp_3pt = 0
        self.cal_bow_warp_3pt()
    def cal_bow_warp_3pt(self):
            data_3pt = pd.read_csv(self.file_path,skiprows=1,names=['x','y','z','x1','x2'])
            #data1 = pd.read_csv(file_path1, skiprows=1,names=['x', 'y', 'z','z1','z2'])
            #data1 = pd.read_csv(file_path1, skiprows=1,names=['x', 'y', 'z'])
            start_loc = 0
            x_ave = np.sum(data_3pt['x']) / len(data_3pt['x'])
            y_ave = np.sum(data_3pt['y']) / len(data_3pt['y'])

            data_3pt['x'] -= x_ave
            data_3pt['y'] -= y_ave
            data_3pt['z'] = -1*data_3pt['z']

            if max(data_3pt['x'])-min(data_3pt['x'])<=150:
                start_loc = 47.5
            else:
                start_loc = 87.5
            #data1 = data1[data1['y']>-67]
            #data1['z'] =1000-data1['z'] 
            #data1['z'] = data1['z']*0.001
            data_3pt['x'] = round(data_3pt['x'],2)
            data_3pt['y'] = round(data_3pt['y'],2)

            data_3pt['dis_to_zero'] = np.sqrt((data_3pt['x'])**2+(data_3pt['y'])**2)
            data_3pt = data_3pt[data_3pt['dis_to_zero']<=start_loc]

            long_rids = start_loc  # 初始位置 
            points_3 = (0,long_rids)
            points_3l = (math.sqrt(3)*long_rids/2,-long_rids/2)
            points_3r = (-math.sqrt(3)*long_rids/2,-long_rids/2)
        

            # #,(long_rids/2,3*math.sqrt(3)*long_rids/2),(-long_rids/2,3*math.sqrt(3)*long_rids/2)] 
            data_3pt['dis_to_top'] = data_3pt.apply(lambda row:math.sqrt((row['x']-points_3[0])**2+(row['y']-points_3[1])**2),axis=1)
            data_3pt['dis_to_left'] = data_3pt.apply(lambda row:math.sqrt((row['x']-points_3l[0])**2+(row['y']-points_3l[1])**2),axis=1)
            data_3pt['dis_to_right'] = data_3pt.apply(lambda row:math.sqrt((row['x']-points_3r[0])**2+(row['y']-points_3r[1])**2),axis=1)

        
            nearest_top = data_3pt.loc[data_3pt['dis_to_top'].idxmin()]
            nearest_left = data_3pt.loc[data_3pt['dis_to_left'].idxmin()]
            nearest_right = data_3pt.loc[data_3pt['dis_to_right'].idxmin()]

            range_threshold = 5 # 圆范围内多少

            data_3pt['dis_to_nearest_top'] = data_3pt.apply(lambda row: math.sqrt((row['x'] - nearest_top['x'])**2 + (row['y'] - nearest_top['y'])**2), axis=1)
            data_3pt['dis_to_nearest_left'] = data_3pt.apply(lambda row: math.sqrt((row['x'] - nearest_left['x'])**2 + (row['y'] - nearest_left['y'])**2), axis=1)
            data_3pt['dis_to_nearest_right'] = data_3pt.apply(lambda row: math.sqrt((row['x'] - nearest_right['x'])**2 + (row['y'] - nearest_right['y'])**2), axis=1)
            

            selected_points = data_3pt[
                (data_3pt['dis_to_nearest_top'] <= range_threshold) |
                (data_3pt['dis_to_nearest_left'] <= range_threshold) |
                (data_3pt['dis_to_nearest_right'] <= range_threshold)
            ].reset_index(drop=True)

            #print(selected_points)

            x = selected_points['x'].values
            y = selected_points['y'].values
            z = selected_points['z'].values


            A = np.column_stack((x, y, np.ones_like(x)))
            coefficients, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            #print(coefficients)
            a, b, c = coefficients
            #data1['new_z'] = -0.022*pow((1- data1['dis_to_zero']/121.72),2.27) + data1['z']

            #data1 = data1[data1['dis_to_zero']<=72]
            #data1 = data1[data1['y']>-67]
            x = data_3pt['x']
            y = data_3pt['y']
            z_ref = data_3pt['z']

            z_ref_min = min(z_ref)
            z_ref_max = max(z_ref)
            x_flat = data_3pt['x'].values.flatten()
            y_flat = data_3pt['y'].values.flatten()
            z_flat = data_3pt['z'].values.flatten()##############################

            # A = np.column_stack((x_flat, y_flat, np.ones_like(x)))
            # coefficients, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
            # a, b, c = coefficients
            #data1['d'] = (a * data1['x'] + b * data1['y'] - data1['z'] + c) / np.sqrt(a**2 + b**2 + 1)
            data_3pt['d'] = (a * x_flat + b * y_flat - z_flat + c) / np.sqrt(a**2 + b**2 + 1)
            data_3pt['d']*=-1
            z_fit = data_3pt['d']

            data_3pt['distances'] = np.abs(a * x + b * y - z_ref + c) / np.sqrt(a**2 + b**2 + 1)
            #print(data1['distances'])
            data_3pt['top_down'] =  a*x+b*y-z_ref+c>0
            #print(data1['distances'])
            # (0,0,z)
            #center_point = data_pos[(data_pos['x']==0)&(data_pos['y']==0)]['distances']

            #TTV
            

            self.Warp_3pt = max(data_3pt[data_3pt['top_down']==True]['distances'])+max(data_3pt[data_3pt['top_down']==False]['distances'])
            #print(len(data1['top_down']==0))
            near_zero_index = data_3pt[data_3pt['dis_to_zero'] <= 71.5]['dis_to_zero'].idxmin()

            Bow_3pt = data_3pt['distances'][near_zero_index]

            if (data_3pt['top_down'][near_zero_index]==False):
                Bow_3pt *= -1
            self.Bow_3pt = -Bow_3pt