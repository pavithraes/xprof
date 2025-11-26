import {Component, ElementRef, EventEmitter, Input, Output, ViewChild, AfterViewInit} from '@angular/core';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';

/** A side navigation component. */
@Component({
  standalone: false,
  selector: 'memory-viewer-control',
  templateUrl: './memory_viewer_control.ng.html',
  styleUrls: ['./memory_viewer_control.scss'],
})
export class MemoryViewerControl implements AfterViewInit {
  private moduleListInternal: string[] = [];

  /** The hlo module list. */
  @Input()
  set moduleList(value: string[]) {
    this.moduleListInternal = value || [];
  }
  get moduleList(): string[] {
    return this.moduleListInternal;
  }

  /** The initially selected module. */
  @Input()
  set firstLoadSelectedModule(value: string) {
    this.selectedModule = value;
  }

  /** The initially selected memory space color. */
  @Input()
  set firstLoadSelectedMemorySpaceColor(value: string) {
    this.selectedMemorySpaceColor = value;
  }

  /** The event when the controls are changed. */
  @Output() readonly changed = new EventEmitter<NavigationEvent>();

  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>;

  selectedModule = '';
  selectedMemorySpaceColor = '';
  filterText = '';

  get filteredModuleList(): string[] {
    if (!this.moduleListInternal) {
      return [];
    }
    if (!this.filterText) {
      return [...this.moduleListInternal];
    }
    const filter = this.filterText.toLowerCase();
    return this.moduleListInternal.filter(
        module => module.toLowerCase().includes(filter));
  }

  ngAfterViewInit() {
    // Defer setting focus on the input to a new task. This is a workaround
    // to prevent an ExpressionChangedAfterItHasBeenCheckedError.
    setTimeout(() => {
      this.searchInput.nativeElement.focus();
    }, 0);
  }

  emitUpdateEvent() {
    this.changed.emit({
      moduleName: this.selectedModule,
      memorySpaceColor: this.selectedMemorySpaceColor,
    });
  }
}
