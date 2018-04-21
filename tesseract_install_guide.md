Download the Teserract setup by clicking [here](https://github.com/arjunvegda/receipt-ocr/blob/master/tesseract_setup.zip "Link to Teserract setup")  

1. Extract anywhere on your pc.

2. There should be 2 directories tessdata and tesseract.

3. Open a project with the opencv settings already configured to run.

4. Move the tessdata directory to the root of your visual studios project that you plan to use.

5. Open the tesseract directory go to the lib folder and find "liblept168.dll" and "libtesseract302.dll" and move them to the project root directory.

6. In visual studios right click on your project and click properties.

7. c/c++>General>Additional Include Directories> point to the include directory inside tesseract

8. Linker>General>Additional Library Directories > point to the lib directory inside tesseract

9. Linker>Input>Additional Dependencies> add these 2 lines "liblept168.lib" "libtesseract302.lib"

You should now be able to use Tesseract by including following directories

```c
#include <tesseract\baseapi.h>
#include <leptonica\allheaders.h>
```

Guide by  - [Rocco Pietrangelo](https://github.com/rpietrangelo "Link to Rocco's Github")
