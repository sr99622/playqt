/*******************************************************************************
* avexception.h
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#ifndef AVEXCEPTION_H
#define AVEXCEPTION_H

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <exception>
#include <iostream>
#include <QObject>

#define NULL_ERROR -666

#define AO2     1
#define AOI     2
#define ACI     3
#define AFSI    4
#define APTC    5
#define APFC    6
#define AWH     7
#define AWT     8
#define AO      9
#define AC     10
#define ACP    11
#define AAOC2  12
#define AFMW   13
#define AFGB   14
#define AHCC   15
#define AFBS   16
#define AWF    17
#define ASP    18
#define ASF    19
#define AEV2   20
#define ARF    21
#define ADV2   22
#define ARP    23
#define AIWF   24
#define AFE    25
#define AAC3   26
#define AFA    27
#define AAC    28
#define AFC    29
#define ABR    30
#define AHFTBN 31
#define AGHC   32
#define ANS    33
#define SGC    34
#define PTF    35
#define GACC   36
#define AFIF   37

using namespace std;

class AVException : public exception
{

public:
    AVException(int id, int tag);
    char error_text[1024];
    int error_id;
    int cmd_tag;

};

class AVExceptionHandler  // : public QObject
{
    //Q_OBJECT

public:
    AVExceptionHandler();
    void ck(int error_id);
    void ck(int ret, int tag);
    QString tag(int cmd_tag);
    QString contextToString(AVCodecContext *arg);
};

#endif // AVEXCEPTION_H
