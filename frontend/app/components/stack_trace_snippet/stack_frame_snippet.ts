import {Component, inject, Input, OnChanges, OnDestroy, SimpleChanges} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Address, Content, SOURCE_CODE_SERVICE_INTERFACE_TOKEN, SourceCodeServiceInterface} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {Subject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/**
 * A component to display a snippet of source code corresponding to a given
 * stack frame address.
 */
@Component({
  standalone: false,
  selector: 'stack-frame-snippet',
  templateUrl: './stack_frame_snippet.ng.html',
  styleUrls: ['./stack_frame_snippet.scss'],
})
export class StackFrameSnippet implements OnChanges, OnDestroy {
  @Input() sourceCodeSnippetAddress: Address|undefined = undefined;
  @Input() topOfStack: boolean|undefined = undefined;
  @Input() usingSourceFileAndLineNumber: boolean|undefined = undefined;
  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly sourceCodeService: SourceCodeServiceInterface =
      inject(SOURCE_CODE_SERVICE_INTERFACE_TOKEN);
  private readonly destroy$ = new Subject<void>();
  private sessionId: string|undefined = undefined;
  frame: Content|undefined = undefined;
  failure: string|undefined = undefined;
  codeSearchLink: string|undefined = undefined;
  codeSearchLinkTooltip: string|undefined = undefined;

  constructor() {
    this.route.params.pipe(takeUntil(this.destroy$)).subscribe((params) => {
      this.sessionId = (params || {})['sessionId'];
      this.reload();
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (this.areDifferentAddresses(
            changes['sourceCodeSnippetAddress']?.previousValue,
            changes['sourceCodeSnippetAddress']?.currentValue)) {
      this.reload();
    }
  }

  trackByIndex(index: number, item: string): number {
    return index;
  }

  get loaded() {
    return this.frame !== undefined || this.failure !== undefined;
  }

  get isAtTopOfStack(): boolean {
    // We intentionally treat `undefined` as `false` here. That is if we don't
    // know whether we are at the top of the stack, we assume that we are not.
    return this.topOfStack ?? false;
  }

  private areDifferentAddresses(
      first: Address|undefined, second: Address|undefined): boolean {
    return first?.fileName !== second?.fileName ||
        first?.lineNumber !== second?.lineNumber ||
        first?.linesBefore !== second?.linesBefore ||
        first?.linesAfter !== second?.linesAfter;
  }

  private reload() {
    this.frame = undefined;
    this.failure = undefined;
    this.codeSearchLink = undefined;
    if (!this.sessionId || !this.sourceCodeSnippetAddress) {
      return;
    }
    this.sourceCodeService
        .loadContent(this.sessionId, this.sourceCodeSnippetAddress)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (frame) => {
            this.frame = frame;
            this.codeSearchLinkTooltip = 'Open in Code Search';
          },
          error: (err) => {
            this.codeSearchLinkTooltip =
                'Try Opening in Code Search (might fail)';
            if (err === null) {
              this.failure = 'Unknown Error';
            } else if ('error' in err && typeof err.error === 'string') {
              this.failure = err.error;
            } else if ('message' in err && typeof err.message === 'string') {
              this.failure = err.message;
            } else {
              this.failure = 'Unknown Error';
            }
          }
        });
    this.sourceCodeService
        .codeSearchLink(
            this.sessionId, this.sourceCodeSnippetAddress.fileName,
            this.sourceCodeSnippetAddress.lineNumber)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (link) => {
            this.codeSearchLink = link;
          },
          error: (err) => {
            console.error('Failed to get code search link', err);
          }
        });
  }
}
