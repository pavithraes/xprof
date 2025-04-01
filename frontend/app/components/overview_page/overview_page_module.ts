import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {OverviewPageBaseModule} from 'org_xprof/frontend/app/components/overview_page/overview_page_base_module';

import {OverviewPage} from './overview_page';

/** An overview page module. */
@NgModule({
  declarations: [OverviewPage],
  imports: [
    CommonModule,
    OverviewPageBaseModule,
  ],
  exports: [OverviewPage]
})
export class OverviewPageModule {
}
