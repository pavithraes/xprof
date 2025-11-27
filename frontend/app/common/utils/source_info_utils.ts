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
 * Parses text content from html string.
 * eg. op name cell in roofline model table, which has a graph viewer link
 * embeded.
 */
export function parseTextContent(htmlString: string): string {
  if (htmlString) {
    const safeHtml = sanitizeHtml(htmlString);
    const doc =
        domParserParseFromString(new DOMParser(), safeHtml, 'text/html');
    const textContentElement = doc.body.firstElementChild as HTMLElement;
    if (textContentElement) {
      return textContentElement.textContent?.trim() || '';
    }
  }
  return '';
}
