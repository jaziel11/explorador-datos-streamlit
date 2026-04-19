import io

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Explorador de Datos", page_icon="📊", layout="wide")

@st.cache_data
def load_data(uploaded_file):
	if uploaded_file is None:
		return None

	filename = uploaded_file.name.lower()
	file_bytes = uploaded_file.getvalue()

	if filename.endswith(".csv"):
		return pd.read_csv(io.BytesIO(file_bytes))

	if filename.endswith(".xlsx") or filename.endswith(".xls"):
		return pd.read_excel(io.BytesIO(file_bytes))

	if filename.endswith(".json"):
		return pd.read_json(io.BytesIO(file_bytes))

	raise ValueError("Formato no soportado. Usa CSV, Excel o JSON.")


def apply_filters(df):
	st.sidebar.subheader("Filtros")
	filtered_df = df.copy()

	columns_to_filter = st.sidebar.multiselect(
		"Selecciona columnas para filtrar",
		options=list(df.columns),
	)

	for col in columns_to_filter:
		series = filtered_df[col]
		st.sidebar.markdown(f"**{col}**")

		if pd.api.types.is_numeric_dtype(series):
			min_val = float(series.min())
			max_val = float(series.max())
			if min_val == max_val:
				st.sidebar.caption("Columna con valor unico; no se aplica filtro.")
				continue

			selected_range = st.sidebar.slider(
				f"Rango de {col}",
				min_value=min_val,
				max_value=max_val,
				value=(min_val, max_val),
			)
			filtered_df = filtered_df[
				filtered_df[col].between(selected_range[0], selected_range[1])
			]

		elif pd.api.types.is_datetime64_any_dtype(series):
			min_date = series.min().date()
			max_date = series.max().date()
			if min_date == max_date:
				st.sidebar.caption("Columna con una sola fecha; no se aplica filtro.")
				continue

			selected_dates = st.sidebar.date_input(
				f"Rango de fechas para {col}",
				value=(min_date, max_date),
				min_value=min_date,
				max_value=max_date,
			)
			if len(selected_dates) == 2:
				start_date, end_date = selected_dates
				filtered_df = filtered_df[
					filtered_df[col].dt.date.between(start_date, end_date)
				]

		else:
			options = sorted(series.dropna().astype(str).unique().tolist())
			selected_values = st.sidebar.multiselect(
				f"Valores de {col}",
				options=options,
				default=options,
			)
			filtered_df = filtered_df[series.astype(str).isin(selected_values)]

	return filtered_df


def try_convert_dates(df):
	converted = df.copy()
	object_cols = converted.select_dtypes(include=["object"]).columns
	for col in object_cols:
		parsed = pd.to_datetime(converted[col], errors="coerce")
		success_ratio = parsed.notna().mean() if len(parsed) else 0
		if success_ratio > 0.7:
			converted[col] = parsed
	return converted


def build_aggregated_chart_data(df, x_axis, y_axis, color_value, aggregation):
	group_columns = [x_axis]
	if color_value and color_value != x_axis:
		group_columns.append(color_value)

	chart_df = df[group_columns + ([y_axis] if y_axis else [])].dropna(subset=group_columns)

	if aggregation == "Conteo":
		aggregated_df = (
			chart_df.groupby(group_columns, dropna=False)
			.size()
			.reset_index(name="valor")
		)
	else:
		aggregation_map = {
			"Suma": "sum",
			"Promedio": "mean",
			"Maximo": "max",
			"Minimo": "min",
		}
		aggregated_df = (
			chart_df.groupby(group_columns, dropna=False)[y_axis]
			.agg(aggregation_map[aggregation])
			.reset_index(name="valor")
		)

	return aggregated_df.sort_values(group_columns).reset_index(drop=True)


def style_correlation_matrix(corr_matrix, strong_threshold=0.7):
	def highlight_strong(val):
		if pd.isna(val) or val == 1:
			return ""
		if val >= strong_threshold:
			return "background-color: rgba(34, 197, 94, 0.45); font-weight: 700;"
		if val <= -strong_threshold:
			return "background-color: rgba(239, 68, 68, 0.45); font-weight: 700;"
		return ""

	return (
		corr_matrix.style
		.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
		.applymap(highlight_strong)
		.format("{:.3f}")
	)


st.title("Explorador interactivo de datos con Streamlit")
st.write(
	"Carga un dataset y explora estadisticas, filtros y visualizaciones interactivas."
)

uploaded_file = st.file_uploader(
	"Sube tu dataset",
	type=["csv", "xlsx", "xls", "json"],
	help="Formatos soportados: CSV, Excel (XLS/XLSX) y JSON.",
)

if uploaded_file is None:
	st.info("Sube un archivo para comenzar.")
	st.stop()

try:
	raw_df = load_data(uploaded_file)
except Exception as exc:
	st.error(f"No se pudo cargar el archivo: {exc}")
	st.stop()

df = try_convert_dates(raw_df)
filtered_df = apply_filters(df)

st.subheader("Vista general")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{len(filtered_df):,}")
col2.metric("Columnas", f"{filtered_df.shape[1]:,}")
missing_values = int(filtered_df.isna().sum().sum())
col3.metric("Valores faltantes", f"{missing_values:,}")

with st.expander("Vista previa de datos", expanded=True):
	st.dataframe(filtered_df, use_container_width=True)

st.subheader("Analisis exploratorio")
tab1, tab2, tab3 = st.tabs(
	["Estadisticas", "Valores faltantes", "Correlaciones"]
)

with tab1:
	st.write("Estadisticas descriptivas")
	st.dataframe(filtered_df.describe(include="all").transpose().T, use_container_width=True)

with tab2:
	na_df = (
		filtered_df.isna().sum().rename("faltantes").to_frame().reset_index(names="columna")
	)
	na_df["porcentaje"] = (na_df["faltantes"] / max(len(filtered_df), 1) * 100).round(2)
	st.dataframe(na_df, use_container_width=True)

with tab3:
	numeric_df = filtered_df.select_dtypes(include="number")
	if numeric_df.shape[1] < 2:
		st.warning("Necesitas al menos 2 columnas numericas para ver correlaciones.")
	else:
		corr_matrix = numeric_df.corr(numeric_only=True)
		styled_corr = style_correlation_matrix(corr_matrix, strong_threshold=0.7)
		st.caption(
			"Colores: verde = correlacion positiva fuerte (>= 0.70), rojo = negativa fuerte (<= -0.70)."
		)
		st.dataframe(styled_corr, use_container_width=True)

st.subheader("Visualizaciones interactivas")

numeric_columns = filtered_df.select_dtypes(include="number").columns.tolist()
all_columns = filtered_df.columns.tolist()

chart_col1, chart_col2, chart_col3, chart_col4, chart_col5 = st.columns(5)
chart_type = chart_col1.selectbox(
	"Tipo de grafico",
	options=["Linea", "Barras", "Dispersion", "Histograma", "Caja"],
)
x_axis = chart_col2.selectbox("Eje X", options=all_columns)

if chart_type in ["Linea", "Barras"]:
	aggregation_options = ["Conteo"]
	if numeric_columns:
		aggregation_options += ["Suma", "Promedio", "Maximo", "Minimo"]
else:
	aggregation_options = ["Sin agregacion"]

aggregation = chart_col3.selectbox("Operacion", options=aggregation_options)

if chart_type in ["Linea", "Barras"]:
	if aggregation == "Conteo":
		y_axis_options = ["Conteo de registros"]
	else:
		y_axis_options = numeric_columns
elif chart_type in ["Dispersion", "Histograma", "Caja"]:
	if numeric_columns:
		y_axis_options = numeric_columns
	else:
		y_axis_options = all_columns
else:
	y_axis_options = all_columns

y_axis = chart_col4.selectbox("Eje Y", options=y_axis_options)
if chart_type in ["Linea", "Barras"]:
	color_label = "Segmentar por"
	color_options = ["Ninguno"] + [col for col in all_columns if col != x_axis]
	color_axis = chart_col5.selectbox(color_label, options=color_options)
	color_value = None if color_axis == "Ninguno" else color_axis
	if aggregation != "Conteo" and not numeric_columns:
		st.warning("No hay columnas numericas disponibles para esta operacion.")
		st.stop()
else:
	color_label = "Color (opcional)"
	color_options = ["Ninguno"] + all_columns
	color_axis = chart_col5.selectbox(color_label, options=color_options)
	color_value = None if color_axis == "Ninguno" else color_axis

if chart_type == "Linea":
	aggregated_df = build_aggregated_chart_data(
		filtered_df,
		x_axis=x_axis,
		y_axis=None if aggregation == "Conteo" else y_axis,
		color_value=color_value,
		aggregation=aggregation,
	)
	st.dataframe(aggregated_df, use_container_width=True)
	st.line_chart(aggregated_df, x=x_axis, y="valor", color=color_value)
elif chart_type == "Barras":
	aggregated_df = build_aggregated_chart_data(
		filtered_df,
		x_axis=x_axis,
		y_axis=None if aggregation == "Conteo" else y_axis,
		color_value=color_value,
		aggregation=aggregation,
	)
	st.dataframe(aggregated_df, use_container_width=True)
	st.bar_chart(aggregated_df, x=x_axis, y="valor", color=color_value)
elif chart_type == "Dispersion":
	st.scatter_chart(filtered_df, x=x_axis, y=y_axis, color=color_value)
elif chart_type == "Histograma":
	if y_axis not in numeric_columns:
		st.warning("Para histograma, selecciona una columna numerica en Eje Y.")
	else:
		st.bar_chart(filtered_df[y_axis].value_counts(bins=30).sort_index())
elif chart_type == "Caja":
	if y_axis not in numeric_columns:
		st.warning("Para diagrama de caja, selecciona una columna numerica en Eje Y.")
	else:
		box_df = filtered_df.dropna(subset=[y_axis]).copy()
		if box_df.empty:
			st.warning("No hay datos suficientes para construir el diagrama de caja.")
		else:
			box_x = x_axis if x_axis != y_axis else None
			if box_x is not None and box_df[box_x].nunique(dropna=True) > 40:
				top_categories = box_df[box_x].astype(str).value_counts().head(40).index
				box_df = box_df[box_df[box_x].astype(str).isin(top_categories)]
				st.caption("Se muestran las 40 categorias mas frecuentes para mejorar legibilidad.")

			box_color = color_value
			if box_color == y_axis:
				box_color = None

			fig_box = px.box(
				box_df,
				x=box_x,
				y=y_axis,
				color=box_color,
				points="outliers",
				title=f"Diagrama de caja de {y_axis}",
			)
			fig_box.update_layout(xaxis_title=box_x or "Muestra", yaxis_title=y_axis)
			st.plotly_chart(fig_box, use_container_width=True)



st.subheader("Exportar resultados")
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
	label="Descargar dataset filtrado en CSV",
	data=csv_data,
	file_name="dataset_filtrado.csv",
	mime="text/csv",
)