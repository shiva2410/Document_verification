import sys
import cv2
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import re
import io
import json
import ftfy
import glob
import random
import math
import tensorflow as tf
from collections import defaultdict
from PIL import ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter
from pancard.opclass import Classifier as oc
import datetime
def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[(N - 1) // 2, :] = 1  # Bug solved with // (integer division)

    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[:, (N - 1) // 2] = 1  # Bug solved with // (integer division)
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0)) / 255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.

    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        dilated_image = np.uint8(dilated_image)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    # print dilation
    # Image.fromarray(edges).show()
    # Image.fromarray(255 * dilated_image).show()
    return contours


def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.
    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        # print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
                print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                    i, covered_sum, new_sum, total, remaining_frac,
                    crop_area(crop), crop_area(new_crop), area, new_area_frac,
                    f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.
    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop

    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im


def process_image(path, out_path):
    orig_im = Image.open(path)
    scale, im = downscale_image(orig_im)

    edges = cv2.Canny(np.asarray(im), 100, 200)

    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges)
    borders.sort(
        key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered

    contours = find_components(edges)
    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    crop = find_optimal_components_subset(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)

    crop = [int(x / scale) for x in crop]  # upscale to the original image size.

    # draw = ImageDraw.Draw(im)
    # c_info = props_for_contours(contours, edges)
    # for c in c_info:
    #    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    #    draw.rectangle(this_crop, outline='blue')
    # draw.rectangle(crop, outline='red')
    # im.save(out_path)
    # draw.text((50, 50), path, fill='red')
    # orig_im.save(out_path)
    # im.show()
    text_im = orig_im.crop(crop)
    text_im.save(path)
    print('%s -> %s' % (path, out_path))

global a 
a=oc()
print("done")
a.init_model()
global graph
graph = tf.get_default_graph()
def pan_validate(image_path):
    global a
    global graph
    image=cv2.imread(image_path)
    print("here2")
    with graph.as_default():
        b=a.get_prediction(image_path)
    print("Predicted:",b)
    if(b=="PAN"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        text = pytesseract.image_to_string(Image.open(filename), lang = 'eng')
    # add +hin after eng within the same argument to extract hindi specific text - change encoding to utf-8 while writing
        os.remove(filename)
    # print(text)
    # show the output images
    # cv2.imshow("Image", image)
    # cv2.imshow("Output", gray)
    # cv2.waitKey(0)
    # writing extracted data into a text file
        text_output = open('outputbase.txt', 'w', encoding='utf-8')
        text_output.write(text)
        text_output.close()
        file = open('outputbase.txt', 'r', encoding='utf-8')
        text = file.read()
    # print(text)
    # Cleaning all the gibberish text
        text = ftfy.fix_text(text)
        text = ftfy.fix_encoding(text)
    
        name = None
        fname = None
        dob = None
        pan = None
        nameline = []
        dobline = []
        panline = []
        text0 = []
        text1 = []
        text2 = []
    # Searching for PAN
        lines = text.split('\n')
        for lin in lines:
            s = lin.strip()
            s = lin.replace('\n','')
            s = s.rstrip()
            s = s.lstrip()
            text1.append(s)
            text1 = list(filter(None, text1))
        lineno=0
        for wordline in text1:
            xx = wordline.split('\n')
            if ([w for w in xx if re.search('(INCOMETAXDEPARWENT @|mcommx|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$', w)]):
                text1 = list(text1)
                lineno = text1.index(wordline)
                break
    # text1 = list(text1)
        text0 = text1[lineno+1:]
        
        def findword(textlist, wordstring):
            lineno = -1
            for wordline in textlist:
                xx = wordline.split( )
                if ([w for w in xx if re.search(wordstring, w)]):
                    lineno = textlist.index(wordline)
                    textlist = textlist[lineno+1:]
                    return textlist
            return textlist
        try:

    # Cleaning first names, better accuracy
            name = text0[0]
            name = name.rstrip()
            name = name.lstrip()
            name = name.replace("8", "B")
            name = name.replace("0", "D")
            name = name.replace("6", "G")
            name = name.replace("1", "I")
            name = re.sub('[^a-zA-Z] +', ' ', name)

        # Cleaning Father's name
            fname = text0[1]
            fname = fname.rstrip()
            fname = fname.lstrip()
            fname = fname.replace("8", "S")
            fname = fname.replace("0", "O")
            fname = fname.replace("6", "G")
            fname = fname.replace("1", "I")
            fname = fname.replace("\"", "A")
            fname = re.sub('[^a-zA-Z] +', ' ', fname)

    # Cleaning DOB
            dob = text0[2]
            dob = dob.rstrip()
            dob = dob.lstrip()
            dob = dob.replace('l', '/')
            dob = dob.replace('L', '/')
            dob = dob.replace('I', '/')
            dob = dob.replace('i', '/')
            dob = dob.replace('|', '/')
            dob = dob.replace('\"', '/1')
            dob = dob.replace(" ", "")

    # Cleaning PAN Card details
            text0 = findword(text1, '(Pormanam|Number|umber|Account|ccount|count|Permanent|ermanent|manent|wumm)$')
            panline = text0[0]
            pan = panline.rstrip()
            pan = pan.lstrip()
            pan = pan.replace(" ", "")
            pan = pan.replace("\"", "")
            pan = pan.replace(";", "")
            pan = pan.replace("%", "L")

        except:
            pass

# Making tuples of data
        data = {}
        data['Name'] = name
        data['Father Name'] = fname
        data['Date of Birth'] = dob
        data['PAN'] = pan
        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str

    # Write JSON file
        with io.open('data.json', 'w', encoding='utf-8') as outfile:
            str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

# Read JSON file
        with open('data.json', encoding = 'utf-8') as data_file:
            data_loaded = json.load(data_file)

        with open('data.json', 'r', encoding= 'utf-8') as f:
            ndata = json.load(f)
        val_score=0
        flag=0
        if(ndata['Name']!=None):
            regex_name = re.compile(r'([a-z]+)( [a-z]+)*( [a-z]+)*$',  re.IGNORECASE) 
            res=regex_name.search(ndata['Name'])
            if(res):
                val_score+=20
        if(ndata['Father Name']!=None):
            regex_name = re.compile('([a-z]+)( [a-z]+)*( [a-z]+)*$',  
              re.IGNORECASE) 
            res=regex_name.search(ndata['Father Name'])
            if(res):
                val_score+=20
        if(ndata['PAN']!=None):
            p = re.compile('[A-Z]{5}[0-9]{4}[A-Z]{1}')
            x=p.findall(ndata['PAN'])
            if(len(x)!=0):
                flag=1
                val_score+=50
                ndata['PAN']=x[0]
        if(ndata['Date of Birth']!=None):
            date_string=ndata['Date of Birth']
            date_format='%d/%m/%Y'
            try:
                date_obj = datetime.datetime.strptime(date_string, date_format)
                ndata['Date of Birth']=str(date_obj)
                val_score+=10
            except ValueError:
                ndata['Date of Birth']=None
        if(flag):
            ndata['result']='valid PAN Card'
            val_per=str(val_score)+'%'
            ndata['score']=val_per
            return ndata
        else:
            ndata['result']='Invalid Pan Card because of invalid pan no'
            val_per=str(val_score)+'%'
            ndata['score']=val_per
            return ndata
    else: 
        ndata={}
        ndata['Name']=None
        ndata['Father Name']=None
        ndata['PAN']=None
        ndata['Date of Birth']=None
        ndata['score']='0%'
        ndata['result']='Not a PAN Card'
        return ndata
        
    

    
