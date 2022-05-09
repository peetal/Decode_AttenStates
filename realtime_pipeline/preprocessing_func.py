import numpy as np 
import os, glob, shutil, time
from subprocess import call
import argparse
from shutil import copyfile
from utils import check_dimension, resample_image



# used during initilization, for processing structrual image

# It is important to set up the correct protocal with scan technicans! Specifically, MPRAGE could output 2 scans, one with 
# distortion correction, one without, we would probably want the distortion corrected version. As a result, we would want the
# scanner to only output one scan, which is the distortion corrected version. 
class Struc:

    def __init__(self, struc_dicom_pattern, cfg):
        """
        dicom_pattern: the pattern of dicom file name used to locate the correct scan series for T1w image
        dicomDir: The directory of all dicom file 
        codeDir: The directory of "realtime_pipeline" folder
        subj_dir: the subject folder
        
        Note: 
            Struc object would only be created when this is the first session for this subject. 
        """
        self.dicom_pattern = struc_dicom_pattern # dicom pattern needs to specify subject ID
        self.dicom_dir = cfg.dicomDir
        self.code_dir = cfg.codeDir
        self.subj_dir = cfg.sub_dir
        self.struc_dir = cfg.struc_dir

    def _mk_struc_dir(self): 
        # this step is not necessary if dcm2niix -n works properly. 
        # But it looks like 2018+ version has bug with it. 
        
        time_start = time.time()
        # make a directory just for structural image
        if not os.path.isdir(os.path.join(self.dicom_dir, "struc")):
            os.mkdir(os.path.join(self.dicom_dir, "struc"))
        else: 
            print("The Struc Dicom directory already exist")
            
        # move dicoms of certain scan number into structural folder 
        struc_dicom = glob.glob(os.path.join(self.dicom_dir, self.dicom_pattern))
        for dicom in struc_dicom: 
            dicom_name = dicom.split('/')[-1]
            shutil.move(dicom, os.path.join(self.dicom_dir, 'struc', dicom_name))
        time_end = time.time()
        time_elapsed = time_end - time_start

        return os.path.join(self.dicom_dir, "struc"), time_elapsed

    def _dcm2nifti(self):

        print("Running dcm2nii for structural DICOM")
        time_start = time.time()
        # get high res T1 
        command = 'dcm2niix -b n -f {0} -o {1} -z y {2}'.format('Struc', self.struc_dir, os.path.join(self.dicom_dir, 'struc'))
        call(command,shell=True)
        #print(command)
        
        # file name 
        struc_nii_filename = os.path.join(self.struc_dir, 'Struc.nii')

        time_end = time.time()
        time_elapsed = time_end - time_start

        return struc_nii_filename, time_elapsed

    def _brain_extraction(self, highres):

        print("Running brain extraction with BET")
        time_start = time.time()
        highres_brain_filename = highres.split('.')[0] + '_brain.nii.gz'
        # use BET for brain extraction 
        command = 'bet {0} {1}'.format(highres, highres_brain_filename)
        call(command,shell=True)

        time_end = time.time()
        time_elapsed = time_end - time_start

        return highres_brain_filename, time_elapsed

    def _brain_segmentation(self, highres_brain):

        """
        Use fast to get highres_wmseg, for faster epi_reg processing. But this is not necessary. 

        Importantly, FAST outputs 3 pve files corresponding to 3 tissue types: 
        pve_0: CSF; pve_1: GM; pve_2: WM. 
        According to https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;1aa3b525.1410 
        If you take the PVE WM image from FAST and run it through fslmaths with -thr 0.5 -bin (thresholding at 0.5 and then binarizing) 
        then you can use the output as the wmseg input for epi_reg or flirt.
        """
        print("Running brain segmentation with FAST")
        time_start = time.time()
        # segmentation 
        command = 'fast {0}'.format(highres_brain)
        call(command,shell=True)

        # select the correct tissue type (pve)
        """
        If used this function, need always to double check whether pve_2 corresponds to WM indeed !! 
        """
        highres_brain_wm = os.path.join(self.struc_dir, 'Struc_brain_pve_2.nii.gz')
        highres_brain_wmseg = os.path.join(self.struc_dir, 'func2struc_fast_wmseg.nii.gz')

        command = 'fslmaths {0} -thr 0.5 -bin {1}'.format(highres_brain_wm, highres_brain_wmseg)
        call(command,shell=True)

        time_end = time.time()
        time_elapsed = time_end - time_start

        return highres_brain_wmseg, time_elapsed

    def run_sub_struc_preprocessing(self):

        time_start = time.time()
        output_str = []
        # organize dicom
        dicom_struc_dir, time_organize = self._mk_struc_dir()
        output_str.append(f"Structual dicoms for this subject: {dicom_struc_dir}, taking {time_organize}s \n")
        
        # dcm2nifti
        highres, time_convert = self._dcm2nifti()
        output_str.append(f"The highres T1w image: {highres}, taking {time_convert}s \n")

        # brain extraction
        highres_brain, time_brain = self._brain_extraction(highres)
        output_str.append(f"The highres T1w brain: {highres_brain}, taking {time_brain}s \n")

        # brain segmentation
        highres_brain_wmseg, time_seg = self._brain_segmentation(highres_brain)   
        output_str.append(f"The highres T1w brain WM seg: {highres_brain_wmseg}, taking {time_seg}s \n")

        time_end = time.time()
        time_elapsed = time_end - time_start
        output_str.append(f"Structrual image preprocessing takes {time_elapsed}s \n")
        
        # write out timing information for processing structural images. 
        struc_file = open(os.path.join(self.struc_dir, "struc_file.txt"),"w")
        struc_file.writelines(output_str)
        struc_file.close()

        return None


# used during inilization, for creating reference bold image on standard space
class RefBOLD:

    def __init__(self, dicomScanNamePattern_refbold, cfg, volume_num=5, linear=False, useFastSeg=True):    

        """
        dicom_pattern: the pattern of dicom file name used to locate the ref bold dicoms
                       This needs to be very specific, e.g.,001_000001_{TR:06d}.dcm
        volume_num: the number of volumes used for computing the ref BOLD 
        dicom_dir: The directory of all dicom file 
        code_dir: The directory of "realtime_pipeline" folder
        sub_dir: subject folder
        refBOLD_temp_dir: temp folder in the refbold directory to store halfway files.
        beh_dir: store timing info for generating refbold images
        sub_roi: store standard2func ROIs
        linear: boolean, linear or nonlinear registration 
        useFastSeg: boolean, whether to use FAST brain segmentation
        roi_list: a list of ROI directories that are on the standard 2mm space. All rois would be transformed on the subj func space

        Importantly: This class assumes that the DICOMs for the ref BOLD image are existed already. 
        i.e., NOT in real time. Aim to implement this in initilization step. 
        """

        self.dicom_pattern = dicomScanNamePattern_refbold
        self.dicom_dir = cfg.dicomDir
        self.code_dir = cfg.codeDir
        self.subj_dir = cfg.sub_dir
        self.roi_list = cfg.roi
        self.sub_roi = cfg.roi_dir
        self.beh_dir = cfg.beh
        self.struc_dir = cfg.struc_dir
        self.refBOLD_dir = cfg.ref_bold
        self.refBOLD_temp_dir = cfg.ref_bold_temp_dir
        self.volumn_num = volume_num
        self.linear = linear
        self.useFastSeg = useFastSeg
        
        # files 
        self.highres_brain = os.path.join(self.struc_dir, 'Struc_brain.nii.gz') # t1_brain
        self.highres_head = os.path.join(self.struc_dir, 'Struc.nii.gz') # t1
        self.highres_wmseg = os.path.join(self.refBOLD_temp_dir, 'func2struc_fast_wmseg.nii.gz') 
        self.standard_brain = os.path.join(self.code_dir, 'data', 'standard', 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz') # MNI_brain
        self.standard_head = os.path.join(self.code_dir, 'data', 'standard', 'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz') # MNI
        self.standard_mask = os.path.join(self.code_dir, 'data', 'standard', 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz') # MNI brain mask

        # files to be created 
        refbold_vol_avg_name = os.path.join(self.refBOLD_dir,'refBOLD_avg')
        self.boldref_avg = refbold_vol_avg_name  # average of all volumes used to construct reference BOLD image. 
        func2struc_mat_name = os.path.join(self.refBOLD_temp_dir,'func2struc.mat')
        self.func2struc_mat = func2struc_mat_name  
        struc2standard_mat_name = os.path.join(self.refBOLD_temp_dir, 'struc2standard.mat')
        self.struc2standard_mat = struc2standard_mat_name
        struc2standard_warp = os.path.join(self.refBOLD_temp_dir, 'struc2standard_warp.nii.gz')
        self.struc2standard_warp = struc2standard_warp
        func2standard_mat_name = os.path.join(self.refBOLD_dir, 'func2standard.mat')
        self.func2standard_mat = func2standard_mat_name
        standard2func_mat_name = os.path.join(self.refBOLD_dir, 'standard2func.mat')
        self.standard2func_mat = standard2func_mat_name
        func2standard_warp_name = os.path.join(self.refBOLD_dir, 'func2standard_warp.nii.gz')
        self.func2standard_warp = func2standard_warp_name
        standard2func_warp_name = os.path.join(self.refBOLD_dir, 'standard2func_warp.nii.gz')
        self.standard2func_warp = standard2func_warp_name
        

    def _dcm2nifti(self):
        
        time_start = time.time()
        for tr in list(range(self.volumn_num)): # for each volume

            full_dicom_name = os.path.join(self.dicom_dir, self.dicom_pattern.format(TR = tr+1))
            if not os.path.isfile(full_dicom_name):
                print(f'{full_dicom_name} does not exist')
                return
            else: 
                print(f'Converting {full_dicom_name}')

            newNiftiToSave = f'refBOLD_{tr}'
            # convert this dicom to nifti, save all dicoms for creating ref_bold together
            command = 'dcm2niix -s y -b n -f {0} -o {1} -z y {2}'.format(newNiftiToSave, self.refBOLD_temp_dir, full_dicom_name)
            call(command,shell=True)

        time_end = time.time()
        time_elapsed = time_end - time_start
        
        return time_elapsed

    def _creat_ref_bold(self):

        time_start = time.time()

        # concatenate volumes on the time domain
        command = 'fslmerge -t {0} {1}'.format(os.path.join(self.refBOLD_temp_dir, 'refBOLD_merge'), os.path.join(self.refBOLD_temp_dir, 'refBOLD*'))
        call(command,shell=True)

        # average
        command = 'fslmaths {0} -Tmean {1}'.format(os.path.join(self.refBOLD_temp_dir,'refBOLD_merge.nii.gz'), os.path.join(self.refBOLD_dir,'refBOLD_avg'))
        call(command,shell=True)

        time_end = time.time()
        time_elapsed = time_end - time_start
        
        return time_elapsed


    def _func2struc(self):

        """
        generating Functional --------> structual transformation matrix. 

        useFastSeg: Whether to use the wmseg file made by Fast, would make things faster. 

        Retruns the name of the full path of func2struc.mat name
        """
        time_start = time.time()

        # generating func2struc.mat

        if self.useFastSeg: 
            # copy wmseg file from to self.refBOLD dir 
            if not os.path.isfile(os.path.join(self.struc_dir, 'func2struc_fast_wmseg.nii.gz')):
                print('WM segmentation file was not found')
                return
            copyfile(os.path.join(self.struc_dir, 'func2struc_fast_wmseg.nii.gz'), os.path.join(self.refBOLD_temp_dir, 'func2struc_fast_wmseg.nii.gz'))
            print("Generating func2struc.mat")
            command = f'epi_reg --epi={self.boldref_avg} --t1={self.highres_head} --t1brain={self.highres_brain} --wmseg={self.highres_wmseg} --out={self.func2struc_mat.split(".")[0]}'
            #print(command)
            call(command,shell=True)
        else: 
            print("Generating func2struc.mat")
            command = f'epi_reg --epi={self.boldref_avg} --t1={self.highres_head} --t1brain={self.highres_brain} --out={self.func2struc_mat.split(".")[0]}'
            call(command,shell=True)

        
        time_end = time.time()
        time_elapsed = time_end - time_start

        return time_elapsed 


    def _struc2standard(self):

        """
        generating structrual --------> standard necessary info 
        Returns elapsed time. 
        """

        time_start = time.time()

        if self.linear: # if use linear registration

            # generating struc2standard.mat
            print("Generating struc2standard.mat")
            command = f'flirt -in {self.highres_brain} -ref {self.standard_brain} -omat {self.struc2standard_mat} -cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp nearestneighbour'
            call(command,shell=True)

            time_end = time.time()
            time_elapsed = time_end - time_start

            return time_elapsed 

        else: # if use nonlinear registration 

            # generating struc2standard.mat 
            print("Generating struc2standard.mat")
            command = f'flirt -in {self.highres_brain} -ref {self.standard_brain} -omat {self.struc2standard_mat} -cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp nearestneighbour'
            call(command,shell=True)

            # generating struc2standard_warp image
            print("Generating struc2standard_warp image")
            command = f'fnirt --in={self.highres_head} --aff={self.struc2standard_mat} --cout={self.struc2standard_warp} --ref={self.standard_head} --refmask={self.standard_mask} --warpres=10,10,10'
            call(command,shell=True)

            time_end = time.time()
            time_elapsed = time_end - time_start

            return time_elapsed 

    def _func2standard(self):

        """
        generating functional --------> standard necessary info 
        Returns elapsed time. 
        """

        print("generating functional --------> standard necessary info ")
        time_start = time.time()

        if self.linear: 

            # generating func2standard.mat 
            
            command = f'convert_xfm -omat {self.func2standard_mat} -concat {self.struc2standard_mat} {self.func2struc_mat}'
            call(command,shell=True)
            
            # generating standard2func.mat, for mapping ROIs to native space 

            command = f'convert_xfm -inverse -omat {self.standard2func_mat} {self.func2standard_mat}'
            call(command,shell=True)
        
        else: 

            # generating func2standard_warp.nii.gz 
            command = f'convertwarp --ref={self.standard_brain} --premat={self.func2struc_mat} --warp1={self.struc2standard_warp} --out={self.func2standard_warp}'
            #print(command)
            call(command,shell=True)
            
            # generating standard2func_warp.nii.gz 
            command = f'invwarp -w {self.func2standard_warp} -o {self.standard2func_warp} -r {self.boldref_avg}'
            call(command,shell=True)
         
        time_end = time.time()
        time_elapsed = time_end - time_start

        return time_elapsed 

    def _ROI_standard2func(self):

        time_start = time.time()
        for roi_nii in self.roi_list:

            roi_name = os.path.join(self.sub_roi, roi_nii.split('/')[-1])
            
            # resample roi if needed 
            if self.linear:  # if linear 
                out_nii = roi_name.split('.')[0] + '_2func_linear.nii.gz'
                command = f'flirt -ref {self.boldref_avg} -in {roi_nii} -out {out_nii} -applyxfm -init {self.standard2func_mat} -interp nearestneighbour'
                call(command,shell=True)
            else: # if nonlinear 
                out_nii = roi_name.split('.')[0] + '_2func_nonlinear.nii.gz'
                command = f'applywarp --ref={self.boldref_avg}  --in={roi_nii} --out={out_nii} --warp={self.standard2func_warp}  --interp=nn'
                call(command,shell=True)
            
        time_end = time.time()
        time_elapsed = time_end - time_start

        return time_elapsed

    def func2standard_register(self):
        
        time_start = time.time()
        output_str = []

        time_dcm2nifti = self._dcm2nifti()
        output_str.append(f'dcm2nifti took {time_dcm2nifti}s \n')
        
        time_boldref = self._creat_ref_bold()
        output_str.append(f'creating boldref took {time_boldref}s \n')
        
        time_func2struc = self._func2struc()
        output_str.append(f'computing func2struc info took {time_func2struc}s \n')
        
        time_struc2standard = self._struc2standard()
        output_str.append(f'computing struc2standard info took {time_struc2standard}s \n')
        
        time_func2standard = self._func2standard()
        output_str.append(f'computing func2standard info {time_func2standard}s \n')

        time_transformROI = self._ROI_standard2func()
        output_str.append(f'Transforming ROI from standard to func took {time_transformROI}s \n')
    
        time_end = time.time()
        time_elapsed = time_end - time_start 
        output_str.append(f'Creating transformation info took {time_elapsed}s')

        refbold_file = open(os.path.join(self.beh_dir, "refbold_file.txt"),"w")
        refbold_file.writelines(output_str)
        refbold_file.close()

        return None

    def apply_transformation_info(self):

        """
        Mainly used for debugging and evaluate registration performance
        by default, the function will transform the ref bol image either lineraly or nonlinearly 
        based on the object constructor. 
        """
        if self.linear:  # if linear 
            out_nii = os.path.join(self.refBOLD_dir, 'refbold_2min_linear.nii.gz')
            command = f'flirt -ref {self.standard_brain} -in {self.boldref_avg} -out {out_nii} -applyxfm -init {self.func2standard_mat} -interp trilinear'
            call(command,shell=True)
        else: # if nonlinear 
            out_nii = os.path.join(self.refBOLD_dir, 'refbold_2min_Nonlinear.nii.gz')
            command = f'applywarp --ref={self.standard_brain}  --in={self.boldref_avg} --out={out_nii} --warp={self.func2standard_warp}'
            call(command,shell=True)
        
        return None


# used in the realtime_pipeline.py script, for real time newNifti2standard space registration. 
class ProcessNewVol:

    def __init__(self, dicom_filename, cfg, linear=False, remove_tempfile=True, mc=True):
        self.dicom_filename = dicom_filename
        self.cfg = cfg
        self.linear = linear # linear registration or nonlinear registration 
        self.remove_tempfile = remove_tempfile
        self.mc = mc
        
    def _convertToNifti(self):

        time_start = time.time()
        # uses dcm2niix to convert incoming dicom to nifti
        # needs to know where to save nifti file output
        # take the dicom file name without the .dcm at the end, and just save that as a .nii
        expected_dicom_name = self.dicom_filename
        nameToSaveNifti = expected_dicom_name.split('.')[0]
        full_dicom_name = '{0}/{1}'.format(self.cfg.dicomDir,expected_dicom_name)

        command = 'dcm2niix -s y -b n -f {0} -o {1} -z y {2}'.format(nameToSaveNifti,self.cfg.raw_bold_dir,full_dicom_name)
        print(command)
        call(command,shell=True)
    

        new_nifti_name = '{0}/{1}.nii.gz'.format(self.cfg.raw_bold_dir,nameToSaveNifti)
        #print(new_nifti_name)
        time_end = time.time()
        time_elapsed = time_end - time_start 

        return new_nifti_name, time_elapsed

    def _motion_correction(self, nifti_name):

        time_start = time.time()
        # use mcflirt for motion correction 
        mc_output_name = nifti_name.split('.')[0].split('/')[-1] + '_mc.nii.gz' 
        mc_output = os.path.join(self.cfg.mc_bold_dir, mc_output_name)
        
        command = f'mcflirt -in {nifti_name} -reffile {self.cfg.boldref_img_native} -out {mc_output}'
        call(command,shell=True)

        #mc_output_nii = mc_output + '.nii.gz'
        time_end = time.time()

        time_elapsed = time_end - time_start
        return mc_output, time_elapsed 

    def _registerToStandard(self, nifti_name):
        
        time_start = time.time()
        # register nifti to standard space
        output_name = nifti_name.split('.')[0].split('/')[-1] + '_standard.nii.gz'
        output_path = os.path.join(self.cfg.mni_bold_dir, output_name)
        
        if self.linear:
            command = f'flirt -ref {self.cfg.standard} -in {nifti_name} -out {output_path} -applyxfm -init {self.cfg.bold2standard_mat} -interp trilinear'
            call(command,shell=True)
        else: 
            command = f'applywarp --ref={self.cfg.MNIstandard_brain}  --in={self.cfg.boldref_img_native} --out={output_path} --warp={self.cfg.func2standard_warp}'
            call(command,shell=True)
        
        time_end = time.time()
        time_elapsed = time_end - time_start

        return output_name, time_elapsed 

    def register_new_nifti(self):

        """
        Given a new DICOM, this function would first convert it into nifti, 
        then perform motion correction, then transform this new nifti from the 
        function space into the standard space using the .mat if linear or 
        .warp if nonlinear. 

        Note that this way would be relatively slow, especially if doing the nonlinear way. 
        A faster method would be keep the new nifti on its function space and apply the 
        transformed ROI directly to the new nifti (the next func. )
        """

        time_start = time.time()
        # convert incoming dicom to nifti
        newNiftiFromDicom, time_nifti2dicom = self._convertToNifti()   
        print(f'New dicom conversion took {time_nifti2dicom}s')  

        if self.mc: 
            # if doing motion correction
            newNifti_mc, mc_time = self._motion_correction(newNiftiFromDicom)
            print(f'Motion correction took {mc_time}s')

            newNifti_reg, registration_time = self._registerToStandard(newNifti_mc)
            print(f'Registration took {registration_time}')

        else: 
            # if not doing motion correction
            newNifti_reg, registration_time = self._registerToStandard(newNiftiFromDicom)
            print(f'Registration took {registration_time}')
        
        if self.remove_tempfile: 
            os.remove(newNiftiFromDicom)
            if self.mc:
                os.remove(newNifti_mc)
       
        time_end = time.time()
        time_elapsed = time_end - time_start
        print(f'Registering this volume took {time_elapsed}s')

        return newNifti_reg

    def dcm2nii_and_mc(self): 

        """
        Given a new DICOM, this function would first convert it into nifti, then perform motion correction, 
        then it would be keep the new nifti on its function space and apply the transformed ROI directly 
        to the new nifti.
        """

        time_start = time.time()
        # convert incoming dicom to nifti
        newNiftiFromDicom, time_nifti2dicom = self._convertToNifti()   
        print(f'New dicom conversion took {time_nifti2dicom}s')  

        if self.mc: 
            # if doing motion correction
            newNifti_mc, mc_time = self._motion_correction(newNiftiFromDicom)
            print(f'Motion correction took {mc_time}s')
        
        if self.remove_tempfile: 
            os.remove(newNiftiFromDicom)
       
        time_end = time.time()
        time_elapsed = time_end - time_start
        print(f'Creating this volume took {time_elapsed}s')

        return newNifti_mc






    
    


            
            

    


        


    
