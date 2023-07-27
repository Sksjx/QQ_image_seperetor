这里是[中文版本](https://github.com/Sksjx/QQ_image_seperetor/blob/main/README.md)

Have you noticed that in the computer's QQ program, there are numerous randomly named folders located in the directory '../AccountName/AccountName/Image/Group2,' and each folder contains randomly named images?

This program is designed for people who don't have a 4090 to run fixed models. It will separate the useless parts from the QQ images and keeps most of the hentai ones.

Here are the steps:

1. In your file manager, locate '../AccountName/AccountName/Image/Group2' and enter an English period '.' in the search bar in the upper right corner, then click 'search.'

2. After the search is complete, scroll down the right scrollbar from top to bottom to find the first image. Click on it once, then drag the right scrollbar to the bottom and press Shift+Left Click on the last image.

3. If you've done the second step correctly, you should have selected all the images in the folder. Cut them with Ctrl+X or copy them with Ctrl+C, then paste them into another folder.

4. Open the program and set the input folder to the folder where you had just pasted the files. Set the output folder to any location you like. On the left, select the number of CPU cores you want to use, and then you can start the process.

Note: The more CPU cores you select, the more computer resources it will consume.

For example, on the author's i7 13700k, selecting 20 cores caused the screen to go black and the program to crash. So it's recommended to choose a suitable number of cores. You can change the number of cores during the program's execution.

For not known reason, when calculating image entropy, running with a single process and 16 processes shows almost the same efficiency. 

If anyone knowledgeable could provide some guidance on where I went wrong with the multiprocessing, i would appreciate a lot.
