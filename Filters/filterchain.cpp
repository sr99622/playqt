/*******************************************************************************
* filterchain.cpp
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

#include "filterchain.h"
#include "mainwindow.h"
#include <chrono>

using namespace std::chrono;

FilterChain::FilterChain(QMainWindow *parent)
{
    mainWindow = parent;
    panel = MW->filterDialog->panel;
}

FilterChain::~FilterChain()
{
}

void FilterChain::process(Frame *vp)
{
    //auto start = high_resolution_clock::now();

    if (!MW->is->paused) {
        fp.copy(vp);
    }
    else {
        Frame *tmp = MW->is->pictq.peek_last();
        if (tmp) {
            if (tmp->frame->width) {
                fp.copy(tmp);
            }
        }
    }

    if (MW->filterDialog->panel->engageFilter->isChecked()) {
        size = panel->leftModel->filters.size();
        for (int i = 0; i < size; i++) {
            panel->leftModel->filters[i]->filter(vp);
        }
    }

    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>(stop - start);
}

