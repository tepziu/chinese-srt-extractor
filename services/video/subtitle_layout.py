"""Fit translated subtitle text to the original hardsub rectangle, not frame bottom."""
from __future__ import annotations

from functools import lru_cache
import unicodedata

from PIL import ImageFont


@lru_cache(maxsize=128)
def _font(size: int):
    for path in ('C:/Windows/Fonts/arialbd.ttf', 'LiberationSans-Bold.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(path,size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _width(text: str, size: int) -> float:
    try:
        font = _font(size)
        measured = float(font.getlength(text))
        if any(unicodedata.east_asian_width(char) in {'W','F'} for char in text):
            # Arial may measure missing CJK glyphs as narrow replacement boxes,
            # while libass correctly falls back to a full-width CJK font.
            conservative = sum(size if unicodedata.east_asian_width(char) in {'W','F'}
                               else float(font.getlength(char)) for char in text)
            measured = max(measured, conservative)
        return measured+max(0,len(text)-1)*.5
    except (ValueError,UnicodeError):
        return sum(size if '\u4e00'<=c<='\u9fff' else size*.58 for c in text)


def _wrap(text: str, size: int, width: int) -> list[str]:
    lines=[]
    for paragraph in text.splitlines() or ['']:
        words=paragraph.split()
        current=''
        for word in words:
            trial=(current+' '+word).strip()
            if _width(trial,size)<=width:
                current=trial;continue
            if current:lines.append(current);current=''
            if _width(word,size)>width:
                # CJK/no-space text and long tokens must still be kept in full.
                piece=''
                for char in word:
                    if piece and _width(piece+char,size)>width:
                        lines.append(piece);piece=''
                    piece+=char
                current=piece
            else:current=word
        if current:lines.append(current)
    return lines or ['']


def fit_subtitle(text: str, width: int, height: int, max_size: int, min_size: int=16) -> dict:
    text=unicodedata.normalize('NFC',str(text).replace('\r','').strip())
    # Leave room for outline/shadow. Never clip/drop words to force a fit.
    usable_width=max(12,width-12);usable_height=max(8,height-8)
    min_size=min(max_size,max(12,min_size))
    last=None
    for size in range(max_size,min_size-1,-1):
        lines=_wrap(text,size,usable_width)
        last={'lines':lines,'font_size':size,'width':max(_width(line,size) for line in lines),
              'height':len(lines)*size*1.15,'overflow':False}
        if len(lines)<=4 and last['height']<=usable_height and last['width']<=usable_width:
            return last
    last['overflow']=True
    return last
