/*******************************************************************************
* filterpanel.cpp
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

#include "filterpanel.h"
#include "mainwindow.h"
#include <QGridLayout>

#include "Filters/subpicture.h"
#include "Filters/darknet.h"

FilterPanel::FilterPanel(QMainWindow *parent) : Panel(parent)
{
    mainWindow = parent;

    filters.push_back(new SubPicture(mainWindow));
    filters.push_back(new Darknet(mainWindow));

    leftView = new FilterListView;
    leftModel = new FilterListModel;
    leftView->setModel(leftModel);
    connect(leftModel, SIGNAL(panelShow(int)), this, SLOT(panelShow(int)));
    connect(leftView, SIGNAL(doubleClick(QMouseEvent*)), this, SLOT(moveRight()));
    connect(leftView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), leftModel, SLOT(onSelectedItemsChanged(QItemSelection, QItemSelection)));

    rightView = new FilterListView;
    rightModel = new FilterListModel;
    for (int i = 0; i < filters.size(); i++) {
        rightModel->filters.push_back(filters[i]);
    }
    rightView->setModel(rightModel);
    connect(rightView, SIGNAL(doubleClick(QMouseEvent*)), this, SLOT(moveLeft()));
    connect(rightView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), rightModel, SLOT(onSelectedItemsChanged(QItemSelection, QItemSelection)));

    moveRightButton = new QPushButton(">");
    moveLeftButton = new QPushButton("<");

    moveUpButton = new QPushButton("^");
    moveUpButton->setMaximumWidth(30);
    moveUpButton->setMinimumHeight(70);
    moveDownButton = new QPushButton("v");
    moveDownButton->setMaximumWidth(30);
    moveDownButton->setMinimumHeight(70);

    connect(moveLeftButton, SIGNAL(clicked()), this, SLOT(moveLeft()));
    connect(moveRightButton, SIGNAL(clicked()), this, SLOT(moveRight()));
    connect(moveUpButton, SIGNAL(clicked()), this, SLOT(moveUp()));
    connect(moveDownButton, SIGNAL(clicked()), this, SLOT(moveDown()));

    QGridLayout *viewLayout = new QGridLayout;
    viewLayout->setAlignment(Qt::AlignCenter);
    QWidget *viewPanel = new QWidget;
    viewLayout->addWidget(moveRightButton,     0, 1, 1, 1, Qt::AlignCenter);
    viewLayout->addWidget(moveLeftButton,      0, 2, 1, 1, Qt::AlignCenter);
    viewLayout->addWidget(moveUpButton,        1, 0, 1, 1);
    viewLayout->addWidget(moveDownButton,      2, 0, 1, 1);
    viewLayout->addWidget(leftView,            1, 1, 4, 1);
    viewLayout->addWidget(rightView,           1, 2, 4, 1);
    viewLayout->setRowStretch(3, 10);
    viewPanel->setLayout(viewLayout);

    tabWidget = new QTabWidget;
    tabWidget->setMinimumHeight(240);
    connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

    engageFilter = new QCheckBox("Engage Filter");
    connect(engageFilter, SIGNAL(stateChanged(int)), this, SLOT(engage(int)));

    QLabel *lbl01 = new QLabel("fps: ");
    fps = new QLabel();

    QLabel *lbl00 = new QLabel("time (ms): ");
    filterTime = new QLabel();

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(viewPanel,        0, 0, 1, 8);
    layout->addWidget(tabWidget,        1, 0, 1, 8);
    layout->addWidget(engageFilter,     2, 0, 1, 1);
    layout->addWidget(lbl01,            2, 2, 1, 1);
    layout->addWidget(fps,              2, 3, 1, 1);
    layout->addWidget(lbl00,            2, 5, 1, 1, Qt::AlignRight);
    layout->addWidget(filterTime,       2, 6, 1, 1, Qt::AlignRight);
    layout->setRowStretch(0, 10);
    setLayout(layout);

    restoreSettings(MW->settings);
}

FilterPanel::~FilterPanel()
{
    while (filters.size() > 0) {
        filters.pop_back();
    }
}

void FilterPanel::autoSave()
{
    for (int i = 0; i < filters.size(); i++) {
        filters[i]->autoSave();
    }
}

Filter *FilterPanel::getCurrentFilter()
{
    Filter *result = nullptr;
    if (leftModel->current_index > -1)
        result = leftModel->filters[leftModel->current_index];
    return result;
}

void FilterPanel::saveSettings(QSettings *settings)
{
    cout << "FilterPanel::saveSettings" << endl;
    for (int i = 0; i < filters.size(); i++) {
        QString arg = "FilterPanel_activeFilter_" + QString::number(i);
        settings->remove(arg);
    }

    for (int i = 0; i < leftModel->filters.size(); i++) {
        QString arg = "FilterPanel_activeFilter_" + QString::number(i);
        settings->setValue(arg, leftModel->filters[i]->name);
    }
}

void FilterPanel::restoreSettings(QSettings *settings)
{
    QStringList list;
    for (int i = 0; i < filters.size(); i++) {
        QString key = "FilterPanel_activeFilter_" + QString::number(i);
        QString value = settings->value(key).toString();
        if (value.length() > 0)
            list.push_back(value);
    }

    for (int i = 0; i < list.size(); i++) {
        for (int j = 0; j < rightModel->filters.size(); j++) {
            if (list[i] == rightModel->filters[j]->name) {
                rightModel->current_index = j;
                moveLeft();
                break;
            }
        }
    }
}

void FilterPanel::idle()
{
    while (leftModel->filters.size() > 0) {
        leftModel->current_index = 0;
        moveRight();
    }
    saveSettings(MW->settings);
}

void FilterPanel::moveLeft()
{
    if (rightModel->filters.size() > 0 && rightModel->current_index > -1) {
        Filter *result = rightModel->filters[rightModel->current_index];
        leftModel->filters.push_back(result);

        tabWidget->addTab(result->panel, result->name);

        leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        leftView->selectionModel()->select(leftModel->setSelectedIndex(leftModel->filters.size()-1), QItemSelectionModel::SelectCurrent);
        rightModel->filters.remove(rightModel->current_index);
        rightModel->emit dataChanged(QModelIndex(), QModelIndex());
        if (rightModel->current_index == rightModel->filters.size()) {
            rightModel->current_index--;
            rightView->selectionModel()->select(rightModel->setSelectedIndex(rightModel->current_index), QItemSelectionModel::SelectCurrent);
        }
    }
    tabWidget->setCurrentIndex(leftModel->current_index);
    saveSettings(MW->settings);
}

void FilterPanel::moveRight()
{
    if (leftModel->filters.size() > 0 && leftModel->current_index > -1) {
        Filter *result = leftModel->filters[leftModel->current_index];

        tabWidget->removeTab(tabWidget->indexOf(result->panel));

        rightModel->filters.push_back(result);
        rightModel->emit dataChanged(QModelIndex(), QModelIndex());
        leftModel->filters.remove(leftModel->current_index);
        leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        if (leftModel->current_index == leftModel->filters.size()) {
            leftModel->current_index--;
            if (leftModel->current_index > -1) {
                leftView->selectionModel()->select(leftModel->setSelectedIndex(leftModel->current_index), QItemSelectionModel::SelectCurrent);
                tabWidget->setCurrentIndex(leftModel->current_index);
            }
            else {
                leftModel->current_index = 0;
            }
        }
    }
    saveSettings(MW->settings);
}

void FilterPanel::moveUp()
{
    if (leftModel->filters.size() > 1 && leftModel->current_index > -1) {
        if (leftModel->current_index > 0) {
            int next_index = leftModel->current_index - 1;
            leftModel->filters.move(leftModel->current_index, next_index);
            leftView->selectionModel()->select(leftModel->setSelectedIndex(next_index), QItemSelectionModel::SelectCurrent);
            leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        }
    }
    saveSettings(MW->settings);
}

void FilterPanel::moveDown()
{
    if (leftModel->filters.size() > 1 && leftModel->current_index > -1) {
        if (leftModel->current_index < leftModel->filters.size()-1) {
            int next_index = leftModel->current_index + 1;
            leftModel->filters.move(leftModel->current_index, next_index);
            leftView->selectionModel()->select(leftModel->setSelectedIndex(next_index), QItemSelectionModel::SelectCurrent);
            leftModel->emit dataChanged(QModelIndex(), QModelIndex());
        }
    }
    saveSettings(MW->settings);
}

void FilterPanel::panelShow(int index)
{
    for (int i = 0; i < tabWidget->count(); i++) {
        if (tabWidget->tabText(i) == leftModel->filters[index]->name) {
            tabWidget->setCurrentIndex(i);
            break;
        }
    }
}

void FilterPanel::tabChanged(int index)
{
    cout << "visible: " << tabWidget->isVisible() << endl;
    cout << "index: " << index << endl;
    cout << "leftModel->current_index: " << leftModel->current_index << endl;
    cout << "leftModel->filters.size(): " << leftModel->filters.size() << endl;

    /*
    if (tabWidget->isVisible() && index > -1) {
        QString text = tabWidget->tabText(index);
        cout << "current text: " << text.toStdString() << endl;
        for (int i = 0; i < leftModel->filters.size(); i++) {
            if (filters[i]->name == text) {
                cout << "found: " << filters[i]->name.toStdString() << endl;
                //leftView->setCurrentIndex(leftView->indexAt(i));
                //leftModel->current_index = i;
                leftView->selectionModel()->select(leftModel->setSelectedIndex(i), QItemSelectionModel::SelectCurrent);
                leftModel->emit dataChanged(QModelIndex(), QModelIndex());
                break;
            }
        }
    }
    */

    /*
    if (tabWidget->isVisible() && index > -1) {
        QString text = tabWidget->tabText(index);
        cout << "current text: " << text.toStdString() << endl;
        for (int i = 0; i < leftModel->filters.size(); i++) {
            if (filters[i]->name == text) {
                leftModel->current_index = i;
                leftView->selectionModel()->select(leftModel->setSelectedIndex(leftModel->current_index), QItemSelectionModel::SelectCurrent);
                leftModel->emit dataChanged(QModelIndex(), QModelIndex());
                break;
            }
        }
    }
    */
}

void FilterPanel::engage(int state)
{
    MW->mainPanel->controlPanel->engageFilter->setChecked(state);
}

bool FilterPanel::isFilterActive(Filter *filter) {
    bool result = false;

    for (int i = 0; i < leftModel->filters.size(); i++) {
        if (leftModel->filters[i]->name == filter->name) {
            result = true;
            break;
        }
    }
    return result;
}

bool FilterPanel::isFilterActive(QString filter_name)
{
    bool result = false;
    for (int i = 0; i < leftModel->filters.size(); i++) {
        if (leftModel->filters[i]->name == filter_name) {
            result = true;
            break;
        }
    }
    return result;
}

Filter *FilterPanel::getFilterByName(QString filter_name)
{
    Filter *result = NULL;
    for (int i = 0; i < filters.size(); i++) {
        if (filters[i]->name == filter_name) {
            result = filters[i];
            break;
        }
    }
    return result;
}

FilterDialog::FilterDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Filter");
    panel = new FilterPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 520;
    defaultHeight = 600;
    settingsKey = "FilterDialog/geometry";
}

FilterPanel *FilterDialog::getPanel()
{
    return (FilterPanel*)panel;
}

void FilterDialog::closeEvent(QCloseEvent *event)
{
    getPanel()->engageFilter->setChecked(false);
    PanelDialog::closeEvent(event);
}

void FilterDialog::keyPressEvent(QKeyEvent *event)
{
    Filter *filter = getPanel()->getCurrentFilter();
    if (filter) {
        getPanel()->getCurrentFilter()->keyPressEvent(event);
    }

    PanelDialog::keyPressEvent(event);
}

void FilterDialog::keyReleaseEvent(QKeyEvent *event)
{
    Filter *filter = getPanel()->getCurrentFilter();
    if (filter) {
        filter->keyReleaseEvent(event);
    }
   PanelDialog::keyReleaseEvent(event);
}
