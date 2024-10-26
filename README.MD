 markdown

###### README #######################################################################################

 ## Description
    A tool for image segmentation by color
    Uses: K-Means clustering algorithm


 ## Features
 - Performs image segmentation using K-Means clustering algorithm
 - Visualize segmented images


###### SETUP ########################################################################################

   ###### 0. Install Necessary Dependencies ###############################################
   To install the necessary dependencies, use the following command:

   ```bash
   pip install -r requirements.txt
   ```

   ###### 1. HOWTO: Ugrade Pip Install in Python Terminal #################################
   PS H:\CODING\PROJECT_IST\ImageSegmentation> python.exe -m pip install --upgrade pip      


   ###### 2. Set Up Virtual Environment ###################################################

   py -m venv ist-env

   .\ist-env\Scripts\Activate.ps1


   ###### 3. Setup Terminals ##############################################################

   #USE TERMINAL: Powershell


   ###### 4. Define Libraries in Terminal #################################################

   ## Install Python Libraries ##
   #Write in terminal: pip install numpy
   #Write in Terminal: pip install cv2
   #Write in terminal: matplotlib

   ## Libraries Not Needed ##
   #Write in Terminal: pip install opencv-python
   #Write in terminal: matplotlib
   #Import CSV library


#### USAGE OF SEGMENTATION TOOL ###################################################################
   To run the segmentation tool, execute the following command:

   ```
   python main1.pynb
   ```


#### Save Changes in Git Repository ###############################################################

   ### **Step 1: Save Your Changes**
   - After editing the README.md file, save your changes (you can use 
   `Ctrl + S` on Windows or `Cmd + S` on Mac).

   ### **Step 2: Commit and Push Changes**
   1. **Open the Terminal** in VS Code.
   2. **Add the README file** to your staging area:
      
   ```bash
   git add README.md
   ```

   3. **Commit the changes** with a meaningful message:

   ```bash
   git commit -m "Add README file"
   ```

   4. **Push the changes** to your GitHub repository:

   ```bash
   git push origin main
   ```