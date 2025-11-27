import {NgZone} from '@angular/core';
import {Chart} from 'org_xprof/frontend/app/components/chart/chart';
import {sanitizeHtml} from 'safevalues';
import {domParserParseFromString} from 'safevalues/dom';

/**
 * Parses source info html string to extract source file, line number and stack
 * trace.
 */
export function parseSourceInfo(sourceInfoHtmlString: string):
    {sourceFileAndLineNumber: string, stackTrace: string} {
  let sourceFileAndLineNumber = '';
  let stackTrace = '';
  if (sourceInfoHtmlString) {
    const safeHtml = sanitizeHtml(sourceInfoHtmlString);
    const doc =
        domParserParseFromString(new DOMParser(), safeHtml, 'text/html');
    const sourceInfoElement = doc.body.firstElementChild as HTMLElement;
    if (sourceInfoElement) {
      sourceFileAndLineNumber = sourceInfoElement.textContent?.trim() || '';
      stackTrace = sourceInfoElement.getAttribute('title')?.trim() || '';
    }
  }
  return {sourceFileAndLineNumber, stackTrace};
}

/**
 * Adds a select listener to chart table.
 */
export function addTableRowSelectListener(
    chartRef: Chart|undefined,
    dataView: google.visualization.DataView|undefined, zone: NgZone,
    handleRowSelection: (
        rowIndex: number,
        rowData: Array<string|number|boolean|Date|null|undefined>) => void) {
  const chart = chartRef?.chart;
  if (!chart) {
    setTimeout(() => {
      addTableRowSelectListener(chartRef, dataView, zone, handleRowSelection);
    }, 100);
    return;
  }
  google.visualization.events.addListener(chart, 'select', () => {
    zone.run(() => {
      const selection = chart.getSelection();
      if (selection && selection.length > 0 && selection[0].row != null) {
        const rowIndex = selection[0].row;
        const rowData: Array<string|number|boolean|Date|null|undefined> = [];
        if (dataView) {
          for (let i = 0; i < dataView.getNumberOfColumns(); i++) {
            rowData.push(dataView.getValue(rowIndex, i));
          }
          handleRowSelection(rowIndex, rowData);
        }
      }
    });
  });
}
