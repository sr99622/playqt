/*******************************************************************************
* filterlistmodel.cpp
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

#include "filterlistmodel.h"
#include <QSlider>
#include <QLabel>

FilterListModel::FilterListModel()
{
}

int FilterListModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return filters.size();
}

QVariant FilterListModel::data(const QModelIndex &index, int role) const
{
    if (index.isValid() && (role == Qt::DisplayRole || role == Qt::EditRole)) {
        Filter *filter = (Filter*)filters[index.row()];
        return filter->name;
    }
    return QVariant();
}

bool FilterListModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (index.isValid() && role == Qt::EditRole) {
        return true;
    }
    else {
        return false;
    }
}

Qt::ItemFlags FilterListModel::flags(const QModelIndex &index) const
{
    return Qt::ItemIsEditable | QAbstractListModel::flags(index);
}

void FilterListModel::beginInsertItems(int start, int end)
{
    beginInsertRows(QModelIndex(), start, end);
}

void FilterListModel::endInsertItems()
{
    endInsertRows();
}

void FilterListModel::onSelectedItemsChanged(QItemSelection selected, QItemSelection deselected)
{
    if (!selected.empty()) {
        current_index = selected.first().indexes().first().row();
        emit panelShow(current_index);
    }
    else {
        current_index = -1;
    }
}

QModelIndex FilterListModel::setSelectedIndex(int row)
{
    QModelIndex index = createIndex(row, 0);
    return index;
}
