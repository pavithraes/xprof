import {AfterViewInit, Component, EventEmitter, inject, Input, NgZone, OnChanges, OnInit, Output, SimpleChanges, ViewChild} from '@angular/core';
import {PIE_CHART_PALETTE} from 'org_xprof/frontend/app/common/constants/roofline_model_constants';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {parseSourceInfo, parseTextContent} from 'org_xprof/frontend/app/common/utils/source_info_utils';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
import {PIE_CHART_OPTIONS, SCATTER_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {Table} from 'org_xprof/frontend/app/components/chart/table/table';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';

type ColumnIdxArr = Array<number|google.visualization.ColumnSpec>;
const SOURCE_INFO_COLUMN_ID = 'source_info';
const PROGRAM_ID_COLUMN_ID = 'hlo_module_id';
const OP_NAME_COLUMN_ID = 'operation';
const OP_CATEGORY_COLUMN_ID = 'category';

/**
 * An operation level analysis table view component (step appregation: total).
 */
@Component({
  standalone: false,
  selector: 'operation-level-analysis',
  templateUrl: './operation_level_analysis.ng.html',
  styleUrls: ['./operation_level_analysis.scss'],
})
export class OperationLevelAnalysis extends Dashboard implements OnInit,
                                                                 OnChanges,
                                                                 AfterViewInit {
  private readonly sourceCodeService =
      inject(SOURCE_CODE_SERVICE_INTERFACE_TOKEN, {optional: true});
  private readonly zone = inject(NgZone);
  /** The roofline model data, original dataset */
  // used for table chart and pie chart
  @Input() rooflineModelData?: google.visualization.DataTable|null = null;
  @Input() viewColumns: ColumnIdxArr = [];
  @Input() sessionId = '';
  // data for scatter chart, heavey data preprocessing handled in parent
  @Input() rooflineSeriesData?: google.visualization.DataTable|null = null;
  @Input() scatterChartOptions: google.visualization.ScatterChartOptions = {};
  // Op name prepopulated from url
  @Input() selectedOp = '';

  @Output()
  readonly filterUpdated =
      new EventEmitter<google.visualization.DataTableCellFilter[]>();

  pieChartDataProvider = new DefaultDataProvider();
  scatterChartDataProvider = new DefaultDataProvider();
  dataInfoCategoryPieChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: {
      ...PIE_CHART_OPTIONS,
      width: 400,
      height: 400,
      chartArea: {
        width: '70%',
        height: '70%',
      },
      title: 'Percentage of self time per HLO op category',
      colors: PIE_CHART_PALETTE,
      sliceVisibilityThreshold: 0.01,
    },
  };
  dataInfoRooflineScatterChart: ChartDataInfo = {
    data: null,
    dataProvider: this.scatterChartDataProvider,
    options: SCATTER_CHART_OPTIONS,
  };

  @ViewChild('table', {read: Table, static: false})
  tableRef: Table|undefined = undefined;
  sourceFileAndLineNumber = '';
  stackTrace = '';
  showStackTrace = false;
  sourceCodeServiceIsAvailable = false;
  // Op info from selected row
  selectedProgramId = '';
  selectedOpName = '';
  selectedOpCategory = '';

  constructor() {
    super();
    this.sourceCodeServiceIsAvailable =
        this.sourceCodeService?.isAvailable() === true;
  }

  ngOnInit() {
    this.update();
  }

  ngAfterViewInit() {
    if (this.sourceCodeServiceIsAvailable) {
      this.addTableRowSelectListener();
    }
  }

  private addTableRowSelectListener() {
    const chart = this.tableRef?.table;
    if (!chart) {
      setTimeout(() => {
        this.addTableRowSelectListener();
      }, 100);
      return;
    }
    google.visualization.events.addListener(chart, 'select', () => {
      this.zone.run(() => {
        const selection = chart.getSelection();
        if (selection && selection.length > 0 && selection[0].row != null) {
          const rowIndex = selection[0].row;
          const rowData: Array<string|number|boolean|Date|null|undefined> = [];
          if (this.dataView) {
            for (let i = 0; i < this.dataView.getNumberOfColumns(); i++) {
              rowData.push(this.dataView.getValue(rowIndex, i));
            }
            this.handleRowSelection(rowIndex, rowData);
          }
        }
      });
    });
  }

  // getter to map from column id to view index, so we can read the value from
  // the selected row.
  // We can remove this and use dataView.getColumnIndex() directly after
  // google.visualization/types is updated to include the interface.
  get tableColumnIdToViewIndexMap(): {[key: string]: number} {
    if (!this.dataView || !this.dataTable) return {};
    const columns = this.dataView.getViewColumns();
    const tableColumnIdToViewIndexMap: {[key: string]: number} = {};
    for (const [index, column] of columns.entries()) {
      tableColumnIdToViewIndexMap[this.dataTable.getColumnId(column)] = index;
    }
    return tableColumnIdToViewIndexMap;
  }

  handleRowSelection(
      rowIndex: number,
      rowData: Array<string|number|boolean|Date|null|undefined>) {
    if (!this.dataView || !this.dataTable) return;
    const programId =
        (rowData[this.tableColumnIdToViewIndexMap[PROGRAM_ID_COLUMN_ID]] as
             string ||
         '').trim();
    // op name cell has a graph viewer link embedded and is an html string.
    const opNameHtmlString =
        (rowData[this.tableColumnIdToViewIndexMap[OP_NAME_COLUMN_ID]] as
             string ||
         '').trim();
    const opName = parseTextContent(opNameHtmlString);
    const opCategory =
        (rowData[this.tableColumnIdToViewIndexMap[OP_CATEGORY_COLUMN_ID]] as
             string ||
         '').trim();

    const sourceInfoHtmlString =
        (rowData[this.tableColumnIdToViewIndexMap[SOURCE_INFO_COLUMN_ID]] as
             string ||
         '').trim();
    this.selectedProgramId = programId;
    this.selectedOpName = opName;
    this.selectedOpCategory = opCategory;

    const {sourceFileAndLineNumber, stackTrace} =
        parseSourceInfo(sourceInfoHtmlString);
    this.sourceFileAndLineNumber = sourceFileAndLineNumber;
    this.stackTrace = stackTrace;
  }

  toggleShowStackTrace() {
    this.showStackTrace = !this.showStackTrace;
  }

  ngOnChanges(changes: SimpleChanges) {
    this.update();
  }

  update() {
    this.parseData();
    // call inheried method to update table chart view
    this.updateView();
  }

  override parseData() {
    // base data already preprocessed in parent component
    if (!this.rooflineModelData) {
      return;
    }

    // process data for table chart
    // columns are used in parent logic to set the dataView
    this.columns = this.viewColumns;
    this.dataTable = this.rooflineModelData;

    // process data for pie chart
    this.pieChartDataProvider.parseData(
        JSON.parse(this.dataTable.toJSON()) as SimpleDataTable,
    );
    this.updateAndDrawPieCharts();

    // process data for roofline scatter chart
    if (this.rooflineSeriesData) {
      this.scatterChartDataProvider.parseData(
          JSON.parse(this.rooflineSeriesData.toJSON()) as SimpleDataTable,
      );
      this.updateAndDrawScatterChart();
    }
  }

  /**
   * Triggered when filter update event is emited
   * this is a temp solutino to make other charts view updated as well as the
   * table chart when filters are changed
   * TODO: remove this function when the Dashboard generalization is done
   * building dashboard with multiple charts
   */
  onUpdateFilters(filter: google.visualization.DataTableCellFilter) {
    this.updateFilters(filter);
    this.updateAndDrawPieCharts();
    this.updateAndDrawScatterChart();
    this.filterUpdated.emit(this.getFilters());
  }

  /**
   * Helper functiont to update data for pie chart and refresh view
   * TODO: update either chart component or Dashboard base class to generalize
   * building dashboard with multiple charts this is a temp solutino to make
   */
  updateAndDrawPieCharts() {
    if (!this.dataTable) return;
    const opCategoryIndex = this.dataTable.getColumnIndex('category');
    const opTotalSelfTimeIndex =
        this.dataTable.getColumnIndex('total_self_time');
    this.dataInfoCategoryPieChart.customChartDataProcessor =
        new CategoryTableDataProcessor(
            this.getFilters(),
            opCategoryIndex,
            opTotalSelfTimeIndex,
        );
  }

  updateAndDrawScatterChart() {
    if (!this.rooflineSeriesData) return;
    this.dataInfoRooflineScatterChart.options = Object.assign(
        {},
        this.dataInfoRooflineScatterChart.options,
        this.scatterChartOptions,
    );
    this.dataInfoRooflineScatterChart.dataProvider.notifyCharts();
  }
}
