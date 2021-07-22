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

FilterChain::FilterChain(QMainWindow *parent)
{
    mainWindow = parent;
    //panel = MW->filterDialog->panel;
}

FilterChain::~FilterChain()
{
}

void FilterChain::process(Frame *vp)
{
    auto start = high_resolution_clock::now();

    if (!counting) {
        t1 = start;
        count = 0;
        counting = true;
    }
    else {
        count++;
    }

    this->vp = vp;

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

    FilterPanel *panel = MW->filterDialog->getPanel();

    if (panel->engageFilter->isChecked()) {
        for (int i = 0; i < panel->leftModel->filters.size(); i++)
            panel->leftModel->filters[i]->filter(vp);
    }

    auto stop = high_resolution_clock::now();
    long msec = duration_cast<milliseconds>(stop - start).count();

    long interval = duration_cast<milliseconds>(stop - t1).count();
    if (interval > 1000) {
        counting = false;
        float fps = 1000 * count / (float)interval;
        if (!k_fps.initialized)
            k_fps.initialize(fps, 0, 0.2f, 0.1f);
        else
            k_fps.measure(fps, interval);
        char buf[64];
        sprintf(buf, "%0.2f", max(k_fps.xh00, 0.0f));
        panel->fps->setText(buf);
    }

    if (!k_time.initialized)
        k_time.initialize(msec, 0, 0.2f, 0.1f);
    else
        k_time.measure(msec, 1);

    char buf[64];
    sprintf(buf, "%0.0f", max(k_time.xh00, 0.0f));
    panel->filterTime->setText(buf);
}

