### IFGL OCR service

This is the OCR service for IFG Life AI Engineer take home task. It lacks several major things to meet the criteria (sorry for that). But, I will point several things I have done on this project below:

```
- Preprocess uploaded image using variance of laplacian and unblur method until it reached
  predefined threshold (default 200, you can see that in app.py)
- Directly infer uploaded and preprocessed image using PaddleOCR
```

Although it was far from what you need in your document spec, it still can extract text from given image. You cannot see the extracted image in REST response, but you can still see the extracted text from application log (shell where app.py is being run).

### Step to run this project

```
shell> python -m venv venv
shell> . venv/bin/activate
(venv) shell> pip install -r requirements.txt
(venv) shell> flask run
```

Once the app is run, you can hit from another shell with command:

```
shell> curl -X POST -F "image=@/path/to/your/image.(png|jpeg|jpg)" http://localhost:5000/extract
```

And if you see the response like this:

```
{"code":200,"message":"All OK"}
```

It means the prediction run smoothly, no single point of error. Then, you can see the extracted text in the shell where the app daemon is run.

NB: You can see downloaded and trained roboflow dataset in ```invoice/```. I use YOLOv8 as its CV training model.
