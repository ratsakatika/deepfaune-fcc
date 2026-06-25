#!/usr/bin/env python3
"""
Fagaras camera-trap dashboard: build one self-contained, interactive HTML file from a
detections CSV (and, optionally, the field-protocol spreadsheet that places the map).

USAGE
    python build_dashboard.py [DETECTIONS.csv] [PROTOCOL.xlsx] [OUTPUT.html]
    defaults: master.csv  FieldProtocols_WTM_FAR_23.xlsx  camera_trap_dashboard.html
    The protocol file is optional; without it the camera-grid map is omitted and the
    rest of the dashboard is built normally.

INPUT CSV must contain the columns:
    filename, date, seqnum, prediction_seq, score_seq, prediction_image,
    score_image, animal_count, human_count
  - `filename` is the full image path; survey scope is read from it (paths under
    'Camera Trap Monitoring/Collections' are the official systematic survey, which is
    the default view; everything else is added by the 'All images' toggle).
  - `date` is parsed as '%Y:%m:%d %H:%M:%S'; out-of-range/unset clocks are dropped from
    the time panels only.
  - The camera/grid id is the WTM_FAR<number> token in `filename`, so the same grid
    cell deployed in different years is merged onto one station.

DETECTION EVENT = one photographic sequence (consecutive frames sharing seqnum in a
    folder), counted once; this de-duplicates bursts (Hurlbert, 1984; Kolowski et al.,
    2021). All figures are sequence-level.

DEPENDENCIES
    pip install pandas numpy plotly openpyxl
  On first run the script downloads Leaflet 1.9.4 and three web fonts and caches them in
  ./dashboard_assets so later runs are offline; set DASHBOARD_ASSET_DIR to relocate it.
  If the downloads fail it falls back to system fonts and a CDN-linked map.

OUTPUT
    One HTML file (~5-6 MB). Plotly, Leaflet, fonts and all data are embedded; only the
    map basemap tiles need a connection when the page is viewed.

PERFORMANCE
    The CSV (~1.8M rows here) is one in-memory pandas pass; every figure is derived twice
    (official survey + all images) and embedded as JSON. No GPU or parallelism is needed
    and peak memory stays well within 32 GB.

References
    Hurlbert, S.H. (1984) 'Pseudoreplication and the design of ecological field
      experiments', Ecological Monographs, 54(2), pp. 187-211. doi:10.2307/1942661.
    Kolowski, J.M. et al. (2021) 'Density and activity patterns ... camera trap data',
      Ecosphere, 12(8), e03350. doi:10.1002/ecs2.3350.
"""
import sys, os, base64, math, colorsys
import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

SRC      = sys.argv[1] if len(sys.argv) > 1 else "master.csv"
PROTOCOL = sys.argv[2] if len(sys.argv) > 2 else "FieldProtocols_WTM_FAR_23.xlsx"
OUT      = sys.argv[3] if len(sys.argv) > 3 else "camera_trap_dashboard.html"

# Embedded assets (fonts + Leaflet) are cached here; first run fetches them, later runs are offline.
ASSET_DIR = os.environ.get("DASHBOARD_ASSET_DIR",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_assets"))
def _asset(name, url):
    """Return bytes for a cached/remote asset, or None on failure. Cache dir: ASSET_DIR."""
    cache = os.path.join(ASSET_DIR, name)
    if os.path.exists(cache):
        return open(cache, "rb").read()
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        b = urllib.request.urlopen(req, timeout=45).read()
        os.makedirs(ASSET_DIR, exist_ok=True); open(cache, "wb").write(b)
        print("  fetched", name, "(%d KB)" % (len(b) // 1024))
        return b
    except Exception as e:
        print("  ! could not obtain", name, "->", e)
        return None

BG="#0c1310"; CARD="#141d18"; EDGE="#243029"; RULE="#2d3b32"; NIGHT2="#0a100d"
INK="#e8efe9"; DIM="#8fa39a"; FAINT="#647269"; NIGHT="rgba(42,54,78,0.42)"
SODIUM="#f0a839"; MOON="#6fb6a6"
GCOL={"ungulates":"#C77D3A","large carnivores":"#B5482B","mesocarnivores":"#D9A441",
 "small mammals":"#8A9740","birds":"#4FA39C","people & livestock":"#8E97A6","unidentified":"#5E6B59"}
GUILD={"wild boar":"ungulates","red deer":"ungulates","roe deer":"ungulates","bison":"ungulates",
 "moose":"ungulates","chamois":"ungulates","ibex":"ungulates","fallow deer":"ungulates","reindeer":"ungulates",
 "bear":"large carnivores","wolf":"large carnivores","lynx":"large carnivores","golden jackal":"large carnivores",
 "fox":"mesocarnivores","mustelid":"mesocarnivores","raccoon dog":"mesocarnivores","cat":"mesocarnivores","raccoon":"mesocarnivores","otter":"mesocarnivores",
 "badger":"mesocarnivores","genet":"mesocarnivores",
 "squirrel":"small mammals","lagomorph":"small mammals","micromammal":"small mammals","hedgehog":"small mammals","porcupine":"small mammals","marmot":"small mammals",
 "bird passerine":"birds","bird galliform":"birds","bird corvid":"birds","bird raptor":"birds",
 "bird columbiform":"birds","bird otherbird":"birds","bird undefined":"birds","bird piciform":"birds","bird anseriform":"birds",
 "human":"people & livestock","vehicle":"people & livestock","dog":"people & livestock",
 "cow":"people & livestock","goat":"people & livestock","equid":"people & livestock","sheep":"people & livestock",
 "undefined":"unidentified"}
GORDER=["ungulates","small mammals","mesocarnivores","large carnivores","birds","people & livestock","unidentified"]
CLOCKS=["ungulates","small mammals","mesocarnivores","large carnivores","birds","people & livestock"]
def guild(p): return GUILD.get(p,"unidentified")
# per-species colour palette (matches the attached design exactly)
SPECIES_COLOUR={
 "bear":"#c77b4e","wolf":"#9fb0bd","lynx":"#f4c542","wild boar":"#b98a52",
 "golden jackal":"#d99a5b","red deer":"#9aa861","roe deer":"#b3bd7e","fallow deer":"#8f9a55",
 "reindeer":"#7e8a52","moose":"#6f7a48","chamois":"#7fa38c","ibex":"#8f9e76","bison":"#a07a4e",
 "squirrel":"#cf9244","fox":"#d2773a","mustelid":"#86a08c","badger":"#9aa0a6",
 "micromammal":"#5f7a6b","lagomorph":"#84957f","hedgehog":"#7a8a72","genet":"#b89a6a",
 "raccoon":"#8a8f96","raccoon dog":"#7f868c","goat":"#7d8a86",
 "bird passerine":"#6fb6a6","bird corvid":"#4f8f84","bird galliform":"#82c0b0",
 "bird raptor":"#5aa392","bird columbiform":"#9ccabe","bird otherbird":"#6aa99b",
 "bird undefined":"#5e9488","bird piciform":"#73b3a4","bird anseriform":"#88c3b6",
 "cow":"#6f7d79","sheep":"#79847f","dog":"#828d88","cat":"#8a938e","equid":"#74807b",
 "porcupine":"#9a8f70","marmot":"#b0a35e","otter":"#6f8f88","human":"#8e97a6","vehicle":"#7a8580"}
def spcol(p): return SPECIES_COLOUR.get(p,"#5c7065")
DOMESTIC={"cow","sheep","dog","cat","equid","goat"}

def shades(base,n):
    base=base.lstrip("#"); r,g,b=[int(base[i:i+2],16)/255 for i in (0,2,4)]
    h,l,s=colorsys.rgb_to_hls(r,g,b)
    if n<=1: return ["#"+base]
    out=[]
    for i in range(n):
        L=0.40+(0.71-0.40)*i/(n-1)
        rr,gg,bb=colorsys.hls_to_rgb(h,L,max(s,0.25))
        out.append("#%02X%02X%02X"%(int(rr*255),int(gg*255),int(bb*255)))
    return out

def distinct(base,n):
    """n visually distinct colours for a guild's species drilldown. The first matches the
    guild hue (so the dominant species reads as the group colour); the rest are spread by
    the golden angle in hue, alternating value/saturation so neighbours separate cleanly on
    the dark background. This replaces same-hue shading, which was hard to tell apart."""
    base=base.lstrip("#"); r,g,b=[int(base[i:i+2],16)/255 for i in (0,2,4)]
    h0,_,_=colorsys.rgb_to_hsv(r,g,b)
    out=[]
    for i in range(n):
        h=(h0+i*0.6180339887)%1.0
        s=0.52 if i%2==0 else 0.70
        v=0.92 if i%2==0 else 0.78
        rr,gg,bb=colorsys.hsv_to_rgb(h,s,v)
        out.append("#%02X%02X%02X"%(int(rr*255),int(gg*255),int(bb*255)))
    return out

# ---------------------------------------------------------------- load + parse
df=pd.read_csv(SRC)
df["dt"]=pd.to_datetime(df["date"],format="%Y:%m:%d %H:%M:%S",errors="coerce")
df.loc[(df["dt"].dt.year<2023)|(df["dt"].dt.year>2025),"dt"]=pd.NaT   # drop epoch/unset-clock dates from time panels
df["recycle"]=df["filename"].str.contains(r"\$RECYCLE\.BIN",regex=True,na=False)
df["station"]=df["filename"].str.extract(r"(WTM_FAR\d+)")[0]
df["inColl"]=df["filename"].str.contains("Camera Trap Monitoring/Collections",na=False,regex=False)
df["prov"]=np.where(df["recycle"],"recycle",np.where(df["station"].notna(),"survey","other"))
df["dirpath"]=df["filename"].str.rsplit("/",n=1).str[0]
df["seq_key"]=df["dirpath"]+"|"+df["seqnum"].astype(str)

ev=(df.sort_values("dt").groupby("seq_key",as_index=False)
      .agg(dt=("dt","first"),pred=("prediction_seq","first"),n_img=("seq_key","size"),
           score=("score_seq","first"),station=("station","first"),prov=("prov","first"),
           recyc=("recycle","first"),inColl=("inColl","first")))

# ---------------------------------------------------------------- theme (scope-independent)
pio.templates["ct"]=go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif",color=DIM,size=13),
    title=dict(font=dict(family="'Space Grotesk', sans-serif",color=INK,size=15)),
    xaxis=dict(gridcolor=RULE,zerolinecolor=RULE,linecolor=EDGE,tickfont=dict(color=DIM),title=dict(font=dict(color=DIM))),
    yaxis=dict(gridcolor=RULE,zerolinecolor=RULE,linecolor=EDGE,tickfont=dict(color=DIM),title=dict(font=dict(color=DIM))),
    legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=DIM,size=11.5)),
    hoverlabel=dict(bgcolor=CARD,bordercolor=EDGE,font=dict(family="Inter, sans-serif",color=INK,size=12)),
    margin=dict(l=56,r=22,t=44,b=44)))
pio.templates.default="ct"

WILD_G=[g for g in GORDER if g not in ("people & livestock","unidentified")]
ALLCOL="#C2A24A"
MLAB=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MINEV=8

def area_fig(x,pivot,colour_map,metaguild=False,hover="%{y}",xaxis_extra=None,ytitle="detection events"):
    fig=go.Figure()
    for col in pivot.columns:
        nm=str(col)[:1].upper()+str(col)[1:]
        tr=go.Scatter(x=x,y=pivot[col].values,mode="lines",name=nm,stackgroup="one",
            line=dict(width=0.8,color=colour_map[col],shape="spline",smoothing=0.7),fillcolor=colour_map[col],
            hovertemplate=hover.replace("{name}",nm)+"<extra></extra>")
        if metaguild: tr.meta=dict(guild=col)
        fig.add_trace(tr)
    GRIDC="#1c2620"
    xa=dict(title=None,gridcolor=GRIDC,zeroline=False,linecolor=GRIDC,tickfont=dict(color=DIM))
    if xaxis_extra: xa.update(xaxis_extra)
    fig.update_layout(height=380,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'IBM Plex Mono', monospace",color=DIM,size=11),
        xaxis=xa,yaxis=dict(title=ytitle,gridcolor=GRIDC,zeroline=False,linecolor=GRIDC,tickfont=dict(color=DIM)),
        legend=dict(orientation="h",yanchor="bottom",y=1.03,x=0,title=None,font=dict(size=10.5)),hovermode="x unified",
        margin=dict(l=54,r=20,t=46,b=34))
    return fig

# ---- camera coordinates from the field protocol (scope-independent)
import re as _re
PROT={}
try:
    import openpyxl as _oxl
    _wb=_oxl.load_workbook(PROTOCOL,data_only=True,read_only=True)
    for _row in _wb["Koordinaten"].iter_rows(values_only=True):
        if not _row or not _row[0]: continue
        _mm=_re.search(r"(WTM_FAR\d+)",str(_row[0]))
        if not _mm: continue
        try: _lat=float(_row[5]); _lon=float(_row[6])
        except (TypeError,ValueError):
            try: _lat=float(_row[3]); _lon=float(_row[4])
            except (TypeError,ValueError): continue
        PROT[_mm.group(1)]={"lat":_lat,"lon":_lon,"setup":_row[7],"time":_row[8],"by":_row[9]}
except Exception:
    PROT={}
HAVE_MAP=len(PROT)>0
FLAGSHIP=["bear","wolf","lynx","wild boar"]
if HAVE_MAP:
    LAT0=float(np.mean([v["lat"] for v in PROT.values()])); LON0=float(np.mean([v["lon"] for v in PROT.values()]))
    _la=[v["lat"] for v in PROT.values()]; _lo=[v["lon"] for v in PROT.values()]
    MAP_BOUNDS={"lat0":round(min(_la),5),"lon0":round(min(_lo),5),"lat1":round(max(_la),5),"lon1":round(max(_lo),5)}
    _sd=pd.to_datetime([v["setup"] for v in PROT.values()],errors="coerce"); SETUP_MIN=_sd.min(); SETUP_MAX=_sd.max()
else:
    LAT0=LON0=0.0; MAP_BOUNDS={}; SETUP_MIN=SETUP_MAX=pd.NaT
MAP_ALLCOL=MOON

# ---------------------------------------------------------------- per-view derivation
def build_view(scope):
    if scope=="survey":
        em=ev["inColl"]; im=df["inColl"]
    else:
        em=(~ev["recyc"]) & ev["station"].notna(); im=(~df["recycle"]) & df["station"].notna()
    sur=ev[em].copy()
    ne=sur[sur["pred"]!="empty"].copy()
    ne["guild"]=ne["pred"].map(guild); ne["stn"]=ne["station"].str.replace("WTM_","",regex=False)
    nd=ne.dropna(subset=["dt"]).copy(); nd["h"]=nd["dt"].dt.hour
    simg=df[im]
    # --- headline statistics
    N_BADDATE=int(simg["dt"].isna().sum())
    N_STN=int(simg["station"].nunique()); N_IMG=len(simg); N_SEQ=len(sur); N_DET=len(ne)
    ACTIVE=int(simg["dt"].dt.date.nunique()); D0,D1=simg["dt"].min(),simg["dt"].max()
    N_TAXA=int(ne["pred"].nunique()); PCT_EMPTY=(100*(sur["pred"]=="empty").mean()) if len(sur) else 0.0
    N_ANIMAL=int(ne[~ne["pred"].isin(["human","vehicle","undefined"])]["pred"].nunique())
    YEARS=(f"{D0:%Y}" if D0.year==D1.year else f"{D0:%Y}\u2013{D1:%Y}") if pd.notna(D0) else ""
    RATIO=round(N_IMG/N_SEQ,1) if N_SEQ else 0
    MED_BURST=int(sur["n_img"].median()) if len(sur) else 0
    MAX_BURST=int(sur["n_img"].max()) if len(sur) else 0
    PCT_IMG_EMPTYSEQ=round(100*sur.loc[sur["pred"]=="empty","n_img"].sum()/sur["n_img"].sum(),0) if len(sur) else 0
    REC_NE=int(((ev["prov"]=="recycle")&(ev["pred"]!="empty")).sum())
    # --- daily-rhythm ring
    def diel24(mask):
        h=nd.loc[mask,"h"]; return [int((h==k).sum()) for k in range(24)]
    sp_counts=ne["pred"].value_counts()
    ring_species=sorted([s for s in sp_counts.index if sp_counts[s]>=MINEV])
    CLOCK_ORDER=["All wildlife"]+ring_species
    CLOCK={}
    for opt in CLOCK_ORDER:
        cnt=diel24(nd["guild"].isin(WILD_G)) if opt=="All wildlife" else diel24(nd["pred"]==opt)
        col=ALLCOL if opt=="All wildlife" else GCOL[guild(opt)]
        CLOCK[opt]={"color":col,"hr":cnt}
    ALL_WILD_N=int(ne[ne["guild"].isin(WILD_G)].shape[0])
    def clab(o): return f"All wildlife ({ALL_WILD_N:,})" if o=="All wildlife" else o[:1].upper()+o[1:]+f" ({int(sp_counts[o]):,})"
    clockOptions=[[o,clab(o)] for o in CLOCK_ORDER]
    # --- season: timeline (weekly) + month-of-year, guild overview + species drilldown
    gcols=[g for g in GORDER if g in nd["guild"].unique()]
    ndw=nd.copy(); ndw["week"]=ndw["dt"].dt.to_period("W").apply(lambda p:p.start_time)
    weeks=list(pd.date_range(ndw["week"].min(),ndw["week"].max(),freq="W-MON")) if len(ndw) else []
    pvT=(ndw.pivot_table(index="week",columns="guild",values="seq_key",aggfunc="size",fill_value=0)
           .reindex(weeks,fill_value=0).reindex(columns=gcols,fill_value=0))
    phenoT_ov=area_fig(weeks,pvT,{g:GCOL[g] for g in gcols},metaguild=True,
        hover="week of %{x|%d %b %Y}<br>%{y} {name} events",ytitle="detection events / week")
    phenoT_dr={}
    for g in gcols:
        sp=ndw[ndw["guild"]==g]["pred"].value_counts(); sp=sp[sp>0].index.tolist()
        if not sp: continue
        pv=(ndw[ndw["guild"]==g].pivot_table(index="week",columns="pred",values="seq_key",aggfunc="size",fill_value=0)
              .reindex(weeks,fill_value=0).reindex(columns=sp,fill_value=0))
        phenoT_dr[g]=area_fig(weeks,pv,dict(zip(sp,distinct(GCOL[g],len(sp)))),
            hover="week of %{x|%d %b %Y}<br>%{y} {name} events",ytitle="detection events / week")
    ndm=nd.copy(); ndm["moy"]=ndm["dt"].dt.month
    def moy_pivot(d,cols): return (d.pivot_table(index="moy",columns=cols,values="seq_key",aggfunc="size",fill_value=0).reindex(range(1,13),fill_value=0))
    pvS=moy_pivot(ndm,"guild").reindex(columns=gcols,fill_value=0)
    phenoS_ov=area_fig(MLAB,pvS,{g:GCOL[g] for g in gcols},metaguild=True,
        hover="%{x}<br>%{y} {name} events",ytitle="detection events / month")
    phenoS_dr={}
    for g in gcols:
        sp=ndm[ndm["guild"]==g]["pred"].value_counts(); sp=sp[sp>0].index.tolist()
        if not sp: continue
        pv=moy_pivot(ndm[ndm["guild"]==g],"pred").reindex(columns=sp,fill_value=0)
        phenoS_dr[g]=area_fig(MLAB,pv,dict(zip(sp,distinct(GCOL[g],len(sp)))),
            hover="%{x}<br>%{y} {name} events",ytitle="detection events / month")
    # --- species bars ("what the cameras saw")
    SB_EXCL={"human","vehicle","undefined"}
    _spev=ne.groupby("pred").size(); _spimg=simg.groupby("prediction_image").size(); _spscore=ne.groupby("pred")["score"].mean()
    SPECIES=[{"species":p,"g":guild(p),"events":int(_spev.get(p,0)),"images":int(_spimg.get(p,0)),
              "mean_score":(round(float(_spscore.get(p)),3) if pd.notna(_spscore.get(p)) else None),
              "dom":(p in DOMESTIC)} for p in _spev.index if p not in SB_EXCL]
    SPECIES.sort(key=lambda d:-d["events"])
    # --- station bars ("busiest cameras")
    _stg=ne.groupby(["station","guild"]).size(); _sttot=ne.groupby("station").size().sort_values(ascending=False)
    def _stmeta(st):
        pr=PROT.get(st)
        if not pr: return {"setup":"","lat":None,"lon":None}
        try: dts=pd.to_datetime(pr["setup"]).strftime("%-d %b %Y")
        except Exception: dts=str(pr.get("setup") or "")
        tm=pr.get("time")
        try: tstr=tm.strftime("%H:%M") if tm is not None else ""
        except Exception: tstr=(str(tm) if tm else "")
        by=str(pr.get("by") or "").strip()
        setup=("Set up "+dts+((" "+tstr) if tstr else "")+((" by "+by) if by else "")).strip()
        return {"setup":setup,"lat":round(pr["lat"],5),"lon":round(pr["lon"],5)}
    STATIONS_BARS=[dict({"id":str(st).replace("WTM_",""),"total":int(_sttot.get(st,0)),
                    "segs":[int(_stg.get((st,g),0)) for g in GORDER]}, **_stmeta(st)) for st in _sttot.index]
    N_SPBAR=len(SPECIES); N_STBAR=len(STATIONS_BARS)
    # --- camera-grid map (positions from protocol; counts from this scope)
    MAP_STATIONS=[]; MAP_SPMAX={}; MAP_MAXTOT=1; SP_GUILD={}; N_MAP=0
    MAP_ORDER=["All wildlife","All wildlife, humans and vehicles"]
    if HAVE_MAP:
        st_sp=ne.groupby(["station","pred"]).size()
        SP_GUILD={p:guild(p) for p in ne["pred"].unique()}
        named=sorted([p for p in ne["pred"].unique() if p!="undefined"])
        seqs_full=simg.groupby("station")["seq_key"].nunique()
        score_by=ne.groupby("station")["score"].mean()
        end_by=simg.groupby("station")["dt"].max()
        for st,info in PROT.items():
            sc={p:int(st_sp.get((st,p),0)) for p in named if st_sp.get((st,p),0)>0}
            wildsc={p:c for p,c in sc.items() if guild(p) in WILD_G}
            tot=int(sum(wildsc.values())); toth=int(tot+sc.get("human",0)+sc.get("vehicle",0))
            top5=[[k,v] for k,v in sorted(sc.items(),key=lambda kv:-kv[1])[:5]]
            try: setup_str=pd.to_datetime(info["setup"]).strftime("%-d %b %Y")
            except Exception: setup_str=str(info["setup"])
            _ed=end_by.get(st,pd.NaT); end_str=(_ed.strftime("%-d %b %Y") if pd.notna(_ed) else None)
            _sv=score_by.get(st,np.nan)
            MAP_STATIONS.append({"id":st.replace("WTM_",""),
                "lat":round(info["lat"],5),"lon":round(info["lon"],5),"setup":setup_str,"end":end_str,
                "seqs":int(seqs_full.get(st,0)),"tot":tot,"toth":toth,"spn":len(sc),"top":top5,
                "score":(round(float(_sv),3) if pd.notna(_sv) else None),"s":sc})
            MAP_MAXTOT=max(MAP_MAXTOT,tot)
        for p in named: MAP_SPMAX[p]=int(max([m["s"].get(p,0) for m in MAP_STATIONS]+[1]))
        MAP_ORDER=["All wildlife","All wildlife, humans and vehicles"]+named; N_MAP=len(MAP_STATIONS)
    def maplabel(o):
        if o=="All wildlife": return f"All wildlife ({sum(m['tot'] for m in MAP_STATIONS):,})"
        if o=="All wildlife, humans and vehicles": return f"All wildlife, humans and vehicles ({sum(m['toth'] for m in MAP_STATIONS):,})"
        n=int(ne[ne['pred']==o].shape[0]); return o[:1].upper()+o[1:]+f" ({n:,})"
    mapOptions=[[o,maplabel(o)] for o in MAP_ORDER]
    # --- clock-error / scope note
    if HAVE_MAP and pd.notna(SETUP_MIN):
        CLOCKERR_NOTE=(f"The field log shows every camera was set up between {SETUP_MIN:%-d %B} and {SETUP_MAX:%-d %B %Y}, "
            f"so records timestamped 2020 or 1970 (unset clocks) are clock errors rather than earlier deployments; "
            f"{N_BADDATE:,} such records are excluded from the time-based panels but still counted in the station and map totals.")
    else:
        CLOCKERR_NOTE=(f"{N_BADDATE:,} records carry timestamps outside the monitoring window or unset-clock (1970/epoch) "
            "values; these are excluded from the time-based panels but still counted in the station totals.")
    # --- KPI strip (largest to smallest) and mutable scalars for header/footer spans
    KPI=[[f"{N_IMG:,}","Images analysed",False],[f"{N_SEQ:,}","Photo sequences",True],
         [f"{N_DET:,}","Animal &amp; human detections",False],[f"{N_STN}","Camera stations",True],
         [f"{N_ANIMAL}","Animal types",False]]
    scalars=dict(N_IMG=f"{N_IMG:,}",N_SEQ=f"{N_SEQ:,}",N_DET=f"{N_DET:,}",N_STN=str(N_STN),N_ANIMAL=str(N_ANIMAL),
                 N_STBAR=str(N_STBAR),N_SPBAR=str(N_SPBAR),ACTIVE=str(ACTIVE),YEARS=YEARS,
                 D0=(D0.strftime('%B %Y') if pd.notna(D0) else ''),D1=(D1.strftime('%B %Y') if pd.notna(D1) else ''),
                 D0d=(D0.strftime('%-d %b %Y') if pd.notna(D0) else ''),D1d=(D1.strftime('%-d %b %Y') if pd.notna(D1) else ''))
    figs=dict(phenoT_ov=phenoT_ov,phenoT_dr=phenoT_dr,phenoS_ov=phenoS_ov,phenoS_dr=phenoS_dr)
    data=dict(SPECIES=SPECIES,STATIONS_BARS=STATIONS_BARS,CLOCK=CLOCK,CLOCK_ORDER=CLOCK_ORDER,clockOptions=clockOptions,
              MAP_STATIONS=MAP_STATIONS,MAP_MAXTOT=MAP_MAXTOT,MAP_SPMAX=MAP_SPMAX,SP_GUILD=SP_GUILD,MAP_BOUNDS=MAP_BOUNDS,
              N_MAP=N_MAP,MAP_ORDER=MAP_ORDER,mapOptions=mapOptions,KPI=KPI,scalars=scalars)
    return dict(data=data,figs=figs,
                N_IMG=N_IMG,N_SEQ=N_SEQ,N_DET=N_DET,N_STN=N_STN,N_ANIMAL=N_ANIMAL,N_TAXA=N_TAXA,ACTIVE=ACTIVE,
                D0=D0,D1=D1,YEARS=YEARS,PCT_EMPTY=PCT_EMPTY,RATIO=RATIO,MED_BURST=MED_BURST,MAX_BURST=MAX_BURST,
                PCT_IMG_EMPTYSEQ=PCT_IMG_EMPTYSEQ,REC_NE=REC_NE,N_BADDATE=N_BADDATE,CLOCKERR_NOTE=CLOCKERR_NOTE,
                N_SPBAR=N_SPBAR,N_STBAR=N_STBAR)

SURV=build_view("survey")
ALLV=build_view("all")
# unpack the official-survey view for the static header/footer text (default scope)
_sv=SURV
N_IMG=_sv["N_IMG"]; N_SEQ=_sv["N_SEQ"]; N_DET=_sv["N_DET"]; N_STN=_sv["N_STN"]; N_ANIMAL=_sv["N_ANIMAL"]; N_TAXA=_sv["N_TAXA"]
ACTIVE=_sv["ACTIVE"]; D0=_sv["D0"]; D1=_sv["D1"]; YEARS=_sv["YEARS"]; PCT_EMPTY=_sv["PCT_EMPTY"]; RATIO=_sv["RATIO"]
MED_BURST=_sv["MED_BURST"]; MAX_BURST=_sv["MAX_BURST"]; PCT_IMG_EMPTYSEQ=_sv["PCT_IMG_EMPTYSEQ"]; REC_NE=_sv["REC_NE"]
N_BADDATE=_sv["N_BADDATE"]; CLOCKERR_NOTE=_sv["CLOCKERR_NOTE"]; N_SPBAR=_sv["N_SPBAR"]; N_STBAR=_sv["N_STBAR"]
N_MAP=SURV["data"]["N_MAP"]
print(f"[survey] stn={N_STN} img={N_IMG} seq={N_SEQ} det={N_DET} animals={N_ANIMAL} span={D0:%d %b %Y}-{D1:%d %b %Y}")
print(f"[all]    stn={ALLV['N_STN']} img={ALLV['N_IMG']} seq={ALLV['N_SEQ']} det={ALLV['N_DET']} animals={ALLV['N_ANIMAL']}")


# ---------------------------------------------------------------- fonts (embedded; fetched once, then cached)
_FONTS=[("Inter",400,"@fontsource/inter@5","inter-latin-400-normal.woff2"),
        ("Inter",500,"@fontsource/inter@5","inter-latin-500-normal.woff2"),
        ("Inter",600,"@fontsource/inter@5","inter-latin-600-normal.woff2"),
        ("Space Grotesk",500,"@fontsource/space-grotesk@5","space-grotesk-latin-500-normal.woff2"),
        ("Space Grotesk",600,"@fontsource/space-grotesk@5","space-grotesk-latin-600-normal.woff2"),
        ("Space Grotesk",700,"@fontsource/space-grotesk@5","space-grotesk-latin-700-normal.woff2"),
        ("IBM Plex Mono",400,"@fontsource/ibm-plex-mono@5","ibm-plex-mono-latin-400-normal.woff2"),
        ("IBM Plex Mono",500,"@fontsource/ibm-plex-mono@5","ibm-plex-mono-latin-500-normal.woff2"),
        ("IBM Plex Mono",600,"@fontsource/ibm-plex-mono@5","ibm-plex-mono-latin-600-normal.woff2")]
faces=[]
for fam,wt,pkg,fn in _FONTS:
    b=_asset(fn,"https://cdn.jsdelivr.net/npm/%s/files/%s"%(pkg,fn))
    if b:
        faces.append("@font-face{font-family:'%s';font-style:normal;font-weight:%d;font-display:swap;src:url(data:font/woff2;base64,%s) format('woff2');}"%(fam,wt,base64.b64encode(b).decode()))
FONT_CSS="".join(faces)
# if embedding failed, link the same families from Google Fonts; CSS stacks fall back to system fonts otherwise
FONT_LINK="" if FONT_CSS else ("<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
    "&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap'>")

# ---------------------------------------------------------------- Leaflet (embedded so markers/popups work offline; only basemap tiles need network)
_LV="1.9.4"; LEAFLET_JS=""; LEAFLET_CSS=""; LEAFLET_HEAD=""; _LJS=""; USE_LEAFLET=False
if HAVE_MAP:
    _ljs=_asset("leaflet.js","https://unpkg.com/leaflet@%s/dist/leaflet.js"%_LV)
    _lcss=_asset("leaflet.css","https://unpkg.com/leaflet@%s/dist/leaflet.css"%_LV)
    if _ljs and _lcss:
        LEAFLET_JS=_ljs.decode("utf-8","ignore"); _css=_lcss.decode("utf-8","ignore")
        for _img in ["layers.png","layers-2x.png","marker-icon.png"]:   # the only images leaflet.css references
            _ib=_asset(_img,"https://unpkg.com/leaflet@%s/dist/images/%s"%(_LV,_img))
            if _ib: _css=_css.replace("images/"+_img,"data:image/png;base64,"+base64.b64encode(_ib).decode())
        LEAFLET_CSS=_css; _LJS="<script>"+LEAFLET_JS+"</script>\n"; USE_LEAFLET=True
    else:   # fallback: link Leaflet from the CDN (the map library then needs a connection too)
        LEAFLET_HEAD="<link rel='stylesheet' href='https://unpkg.com/leaflet@%s/dist/leaflet.css'>"%_LV
        _LJS="<script src='https://unpkg.com/leaflet@%s/dist/leaflet.js'></script>\n"%_LV; USE_LEAFLET=True

CSS="""
:root{
 --night:#0c1310; --night2:#0a100d; --bark:#141d18; --bark2:#18241e;
 --line:#243029; --line2:#2d3b32;
 --mist:#e8efe9; --muted:#8fa39a; --faint:#647269;
 --sodium:#f0a839; --moon:#6fb6a6; --empty:#2c352f; --r:14px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%}
body{background:radial-gradient(1200px 600px at 80% -10%, #12201a 0%, transparent 60%),radial-gradient(900px 500px at -10% 10%, #101a15 0%, transparent 55%),var(--night);color:var(--mist);font-family:Inter,system-ui,-apple-system,sans-serif;line-height:1.5;letter-spacing:.005em;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 70px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;font-weight:600;margin:0}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--sodium);margin-bottom:14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.eyebrow .pdot{width:6px;height:6px;border-radius:50%;background:var(--sodium);box-shadow:0 0 10px 2px var(--sodium);animation:pulse 2.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
h1{font-size:clamp(30px,5vw,50px);line-height:1.03;letter-spacing:-.015em;margin-bottom:0}
h1 .em{color:var(--sodium)}
.thesis{color:var(--muted);max-width:62ch;margin-top:14px;font-size:15px}
.thesis b{color:var(--mist);font-weight:600}
.kpis{margin-top:26px;display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
.kpi{background:var(--bark);border:1px solid var(--line);border-radius:var(--r);padding:17px 16px;position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--sodium);opacity:.55}
.kpi.alt::before{background:var(--moon)}
.kpi .v{font-family:'IBM Plex Mono',monospace;font-size:25px;font-weight:500;letter-spacing:-.02em;line-height:1}
.kpi .v small{color:var(--muted);font-size:12px}
.kpi .k{font-size:11px;color:var(--muted);margin-top:7px;letter-spacing:.03em}
.section{margin-top:42px}
.section-head{display:flex;align-items:baseline;gap:14px;margin-bottom:16px}
.section-head .n{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--sodium);padding-top:2px}
.section-head h2{font-size:21px;letter-spacing:-.01em}
.section-head .hint{color:var(--faint);font-size:12.5px;margin-left:auto;max-width:50ch;text-align:right}
.lead{color:var(--muted);max-width:82ch;font-size:13.5px;margin:-2px 0 15px}
.lead b{color:var(--mist);font-weight:600}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.controls .lbl{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--faint);letter-spacing:.04em}
.controls .cap{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--muted)}
.toggle{display:inline-flex;border:1px solid var(--line2);border-radius:8px;overflow:hidden}
.scopebar{display:flex;align-items:center;gap:12px;margin:14px 0 22px;flex-wrap:wrap}
.scopebar .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em}
.scopebar .cap{color:var(--faint);font-size:12px}
.toggle button{background:transparent;border:none;color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:11px;padding:6px 12px;cursor:pointer;letter-spacing:.03em}
.toggle button.on{background:var(--sodium);color:#1a1206;font-weight:600}
.backbtn{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--mist);background:var(--night2);border:1px solid var(--line2);border-radius:8px;padding:6px 12px;cursor:pointer;letter-spacing:.03em}
.backbtn:hover{background:#1c2a23}
select{background:var(--night2);color:var(--mist);border:1px solid var(--line2);border-radius:8px;padding:7px 28px 7px 11px;font-family:'IBM Plex Mono',monospace;font-size:12px;cursor:pointer;-webkit-appearance:none;appearance:none;background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6'><path d='M0 0l5 6 5-6z' fill='%238fa39a'/></svg>");background-repeat:no-repeat;background-position:right 10px center}
select:focus-visible,button:focus-visible{outline:2px solid var(--moon);outline-offset:2px}
.card{background:var(--bark);border:1px solid var(--line);border-radius:var(--r);padding:20px}
.sbar{display:flex;align-items:center;gap:12px;padding:5px 0}
.sbar .nm{width:128px;flex:none;font-size:12.5px;text-align:right;color:var(--mist);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sbar .track{flex:1;height:18px;background:#0e1612;border-radius:5px;overflow:hidden;position:relative}
.sbar .fill{height:100%;border-radius:5px;transition:width .6s cubic-bezier(.2,.7,.2,1)}
.sbar .val{width:64px;flex:none;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);text-align:right}
.sbar:hover .fill{filter:brightness(1.18)}
.stbar{display:flex;align-items:center;gap:10px;padding:4px 0;font-size:12px}
.stbar .nm{width:120px;flex:none;text-align:right;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'IBM Plex Mono',monospace;font-size:11px}
.stbar .track{flex:1;height:16px;background:#0e1612;border-radius:4px;overflow:hidden;display:flex}
.stbar .seg{height:100%}
.stbar .val{width:46px;flex:none;text-align:right;font-family:'IBM Plex Mono',monospace;color:var(--muted)}
.glegend{display:flex;gap:16px;flex-wrap:wrap;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);margin:0}
.glegend span{display:inline-flex;align-items:center;gap:6px}
.glegend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.glegend .gl{cursor:pointer;transition:opacity .12s,color .12s}
.glegend .gl:hover{color:var(--mist)}
.dielring{display:block;margin:2px auto 0;width:100%;max-width:330px;height:auto}
.ring-legend{display:flex;gap:18px;margin:0 0 6px;font-size:11px;color:var(--faint);justify-content:center}
.ring-legend span{display:inline-flex;align-items:center;gap:6px}
.ring-legend i{width:18px;height:3px;border-radius:2px;display:inline-block}
.spoke{cursor:pointer;transition:stroke-width .08s}
#fig_M{height:520px;padding:0;overflow:hidden;border:1px solid var(--line);border-radius:var(--r);background:var(--night2)}
.leaflet-container{background:var(--night2);font-family:'Inter',sans-serif;font-size:12px}
.leaflet-container a{color:var(--moon)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:#0a120e;color:var(--mist);border:1px solid var(--line2);box-shadow:0 6px 24px rgba(0,0,0,.5)}
.leaflet-popup-content{margin:12px 14px;font-family:'Inter',sans-serif;font-size:12.5px;line-height:1.55}
.leaflet-popup-content .pid{font-family:'IBM Plex Mono',monospace;font-weight:600;color:var(--sodium);font-size:13px;margin-bottom:5px}
.leaflet-popup-content .stats{display:grid;grid-template-columns:1fr 1fr;column-gap:18px;row-gap:3px;margin:3px 0}
.leaflet-popup-content .row{display:flex;justify-content:space-between;gap:10px;color:var(--muted);white-space:nowrap}
.leaflet-popup-content .row b{font-family:'IBM Plex Mono',monospace;color:var(--mist);font-weight:500}
.leaflet-popup-content .sps{margin-top:9px;color:var(--muted)}
.leaflet-popup-content .pri{margin-top:9px;display:flex;gap:5px 6px;flex-wrap:wrap;align-items:center}
.leaflet-popup-content .pri .plbl{color:var(--muted);margin-right:2px}
.leaflet-popup-content .chip{font-family:'IBM Plex Mono',monospace;font-size:10.5px;padding:1px 6px;border-radius:5px;background:#15201a;white-space:nowrap}
.leaflet-bar a{background:var(--bark);color:var(--mist);border-color:var(--line2)}
.leaflet-bar a:hover{background:#1c2a23}
.leaflet-control-layers{background:var(--bark);color:var(--mist);border:1px solid var(--line2);border-radius:8px}
.leaflet-control-layers-expanded{padding:8px 10px 8px 8px}
.leaflet-control-attribution{background:rgba(10,18,14,.78)!important;color:var(--faint)!important}
#tip{position:fixed;pointer-events:none;background:#050907;border:1px solid var(--line2);border-radius:8px;padding:8px 13px;font-size:12px;opacity:0;transition:opacity .12s;z-index:70;max-width:420px;color:var(--mist)}
#tip .t{font-family:'Space Grotesk',sans-serif;font-weight:600;margin-bottom:4px;white-space:nowrap}
#tip .sub{color:var(--muted);font-size:11px;margin:-1px 0 4px;white-space:nowrap}
#tip .sub:last-of-type{margin-bottom:6px}
#tip .rs{display:flex;flex-wrap:wrap;gap:3px 16px;color:var(--muted)}
#tip .rs span{white-space:nowrap}
#tip .rs b{font-family:'IBM Plex Mono',monospace;color:var(--mist);font-weight:500}
footer{margin-top:50px;border-top:1px solid var(--line);padding-top:22px;font-size:12.5px;color:var(--faint);max-width:90ch}
footer h3{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
footer p{margin-bottom:9px}footer b{color:var(--muted);font-weight:600}
.hint{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint)}
html,body{max-width:100%;overflow-x:hidden}
@media (max-width:760px){
 .kpis{grid-template-columns:repeat(2,1fr)}
 .section-head .hint{display:none}
 .sbar .nm{width:96px} .stbar .nm{width:86px}
 #fig_M{height:400px}
 .wrap{padding:26px 14px 60px}
 .card{padding:14px}
}
"""

head=(
"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
"<meta name='viewport' content='width=device-width, initial-scale=1'>"
"<title>Camera-trap survey \u00b7 F\u0103g\u0103ra\u015f Mountains</title>"
f"{FONT_LINK}{LEAFLET_HEAD}<style>{FONT_CSS}{LEAFLET_CSS}{CSS}</style></head><body><div class='wrap'>"
f"<div class='eyebrow'><span class='pdot'></span>Wildlife camera-trap survey \u00b7 F\u0103g\u0103ra\u015f Mountains, Romania \u00b7 <span id='th_years'>{YEARS}</span></div>"
f"<h1><span id='th_stn'>{N_STN}</span> cameras across the<br><span class='em'>F\u0103g\u0103ra\u015f Mountains</span>.</h1>"
"<div class='scopebar'><span class='lbl'>Show</span>"
"<div class='toggle' id='scopeTog'>"
"<button data-v='survey' class='on' onclick=\"setScope('survey')\">Official survey</button>"
"<button data-v='all' onclick=\"setScope('all')\">All images</button></div>"
"<span class='cap' id='scopeNote'></span></div>"
f"<p class='thesis'>Spread across a stretch of the Southern Carpathians in Romania, the cameras recorded "
f"<b id='th_img'>{N_IMG:,} photographs</b> between <b id='th_d0'>{D0:%B %Y}</b> and <b id='th_d1'>{D1:%B %Y}</b>. Most frames were empty; the rest "
f"hold <b id='th_det'>{N_DET:,} animal and human detections</b> across <span id='th_animal'>{N_ANIMAL}</span> kinds of animal, from squirrels and roe deer "
f"to bear, wolf and bison. This dashboard shows what moved through these mountains, where, and when.</p>"
"<div class='kpis' id='kpis'></div>")

def section(num,title,lead,div_id,controls="",below=""):
    ctl=f"<div class='controls'>{controls}</div>" if controls else ""
    return (f"<section class='section'><div class='section-head'><span class='n'>{num}</span>"
            f"<h2>{title}</h2></div><p class='lead'>{lead}</p>{ctl}"
            f"<div class='card' id='{div_id}'></div>{below}</section>")

sb_legend='<div class="glegend" id="sbLegend"></div>'
st_legend='<div class="glegend" id="stLegend"></div>'
# Render the default-view (survey) options into the static HTML so every option, including
# "All wildlife, humans and vehicles", is present on first paint; JS only refills on scope switch.
def _opts_html(options,sel="All wildlife"):
    return "".join('<option value="%s"%s>%s</option>'%(str(v).replace("&","&amp;").replace('"',"&quot;"),
        (" selected" if v==sel else ""),lab) for v,lab in options)
map_opts=_opts_html(SURV["data"]["mapOptions"])
clock_opts=_opts_html(SURV["data"]["clockOptions"])

panels=[]
if HAVE_MAP and USE_LEAFLET:
    panels.append(("The camera grid",
        f"The {N_MAP} cameras at their real positions. Each circle is a camera; choose \u2018All wildlife\u2019, "
        "\u2018All wildlife, humans and vehicles\u2019 or a single species to size the circles by how many detections each camera "
        "made, with empty cameras dimmed. Tap a camera for its details. If the map background does not appear, pick a "
        "different style from the control at the top right; the camera markers always work.",
        "fig_M",
        f'<span class="lbl">Show</span><select id="mapSel" onchange="buildMap(this.value)">{map_opts}</select>',
        '<div id="mapNote" class="hint" style="margin-top:12px"></div>'))
panels.append(("Daily rhythm",
    "When animals are active, by hour of day. Each spoke is one hour and its length is the number of detections in "
    "that hour; midnight is at the top, noon at the bottom, and the shaded half is night (18:00\u201306:00). The centre "
    "shows the total for the current choice. Only records carrying a usable time of day are counted here, so these "
    "totals are smaller than the survey-wide figures at the top of the page. Choose \u2018All wildlife\u2019 or a single "
    "species; each ring is scaled to its own busiest hour.",
    "fig_A",
    f'<span class="lbl">Show</span><select id="clockSel" onchange="buildClock(this.value)">{clock_opts}</select><span id="clockN" class="cap"></span>',
    ""))
panels.append(("Through the seasons",
    "Detection events stacked by ecological group. <b>Timeline</b> shows them week by week across the survey window; "
    "<b>By month</b> folds the years onto a single calendar (month of year) so the seasonal pattern stands out. "
    "Effort is uneven (cameras ran on different dates), so totals reflect both activity and how many cameras were "
    "running. Click a group in the legend to break it down by species.",
    "fig_D",
    '<div class="toggle" id="seasonTog"><button data-m="T" class="on" onclick="setSeasonMode(\'T\')">Timeline</button>'
    '<button data-m="S" onclick="setSeasonMode(\'S\')">By month</button></div>'
    '<button id="back_D" class="backbtn" style="display:none" onclick="back(\'D\')">&larr;&nbsp;back to groups</button>',
    ""))
panels.append(("What the cameras saw",
    "Every species the cameras recorded, as detection events (one per photographic sequence) or as photographs. "
    "All taxa are shown, ordered by frequency and coloured by ecological group (matching the season and station "
    "panels); domestic animals are muted. Counts are sequence-level, so they differ from a frame-by-frame tally "
    "(see notes).",
    "speciesBars",
    '<div class="toggle" id="sbToggle"><button data-m="events" class="on">Events</button>'
    '<button data-m="images">Photos</button></div>'+sb_legend,
    '<div class="hint" style="margin-top:12px">Domestic animals (cattle, dogs, sheep, goats, cats, horses) are shown muted. Click a group in the legend to hide or show it; hover a bar for events, photos and mean confidence.</div>'))
panels.append(("Busiest cameras",
    f"Every camera station, ordered by total detection events, each bar split by ecological group. All {N_STBAR} "
    "stations are shown. Bar length is the number of detection events (one per photographic sequence); click a group "
    "in the legend to hide or show it, and hover a bar for the station total.",
    "stationBars",st_legend,""))
body="".join(section(f"{i+1:02d}",*p) for i,p in enumerate(panels))

footer=(
"<footer><h3>Notes &amp; method</h3>"
"<p><b>Unit of analysis.</b> Consecutive frames sharing a sequence number within a folder form one photographic "
"sequence (burst); each sequence counts as one detection event, since burst frames are not independent records "
"of separate visits.</p>"
f"<p><b>Why ~{RATIO:g} images per sequence, not three.</b> A typical trigger is a short burst: the median "
f"sequence is {MED_BURST} frames. The mean is about {RATIO:g} only because a long tail of sequences fired "
f"repeatedly (an animal lingering in front of the camera, or wind and vegetation), up to {MAX_BURST:,} frames in "
f"a single case; about {PCT_IMG_EMPTYSEQ:.0f}% of all frames sit in empty sequences. Counting events per "
"sequence rather than per image removes this inflation.</p>"
f"<p><b>Events versus frames.</b> Every figure counts detection events (one per sequence) using the "
"sequence-level prediction. This differs by design from a frame-level tally such as an Excel pivot of "
"prediction_image, which counts every photograph and is dominated by long bursts; the two were checked and "
"the frame-level totals reconcile exactly with that pivot.</p>"
f"<p><b>Scope.</b> The default view is the official systematic survey: images in the drive's "
f"<i>Camera Trap Monitoring/Collections</i> folder, covering <span id='ft_stn'>{N_STN}</span> stations, "
f"<span id='ft_img'>{N_IMG:,}</span> images, <span id='ft_seq'>{N_SEQ:,}</span> sequences and "
f"<span id='ft_active'>{ACTIVE}</span> active days (<span id='ft_d0'>{D0:%d %b %Y}</span>\u2013"
f"<span id='ft_d1'>{D1:%d %b %Y}</span>). The <i>All images</i> toggle additionally counts off-protocol "
f"distance-calibration frames recorded at the same cells (a few hundred images, mostly empty). Deleted files in "
f"the drive's recycle bin ({REC_NE} non-empty records) are excluded from both views, as their provenance and "
f"effort are undefined.</p>"
"<p><b>Caveats.</b> Labels are automated predictions and are not field-verified; the 'unidentified' group is "
"large and clusters in daylight, consistent with empty or vegetation triggers. The seasonal panel is not "
"effort-corrected. Night bands are nominal, not computed from local sunrise/sunset. "
f"Time-based panels (the clock and the season chart) cover the main monitoring window, {D0:%B %Y} to "
f"{D1:%B %Y}. {CLOCKERR_NOTE} "
"Drilldowns and the clock for sparse groups rest on few events "
"and are descriptive only.</p>"
+ (f"<p><b>Place.</b> The study area is the F\u0103g\u0103ra\u015f Mountains (Southern Carpathians, Romania); the "
   f"study-area code FAR denotes F\u0103g\u0103ra\u015f. The {N_MAP} cameras lie around {LAT0:.3f}\u00b0N, {LON0:.3f}\u00b0E, "
   "with coordinates and setup dates taken from the 2023 field protocol. The survey is a fixed grid re-deployed "
   "across years: every camera (2023 and 2024) is placed from the 2023 protocol by matching grid number and "
   "ignoring the year suffix, and every survey cell has a protocol match.</p>" if HAVE_MAP else "")
+ f"<p style='color:#5E6B59;margin-top:14px'>Sources: master.csv ({len(df):,} image records)"
+ (f" and {os.path.basename(PROTOCOL)} (camera coordinates)" if HAVE_MAP else "")
+ (" \u00b7 built with pandas, Plotly and Leaflet. Data, fonts and code are embedded in this single file; only the "
   "map background needs an internet connection, and the camera markers work without one." if USE_LEAFLET else
   " \u00b7 built with pandas + Plotly \u00b7 self-contained, no external resources.")
+ "</p></footer></div>")

# ---------------------------------------------------------------- client JS (build figures + drilldown + inventory toggling)
import json as _json
def J(fig): return fig.to_json()
def _figs_js(fg):
    return ("{phenoT_ov:"+J(fg["phenoT_ov"])+",phenoS_ov:"+J(fg["phenoS_ov"])
            +",phenoT_dr:{"+",".join('"%s":%s'%(g,J(f)) for g,f in fg["phenoT_dr"].items())+"}"
            +",phenoS_dr:{"+",".join('"%s":%s'%(g,J(f)) for g,f in fg["phenoS_dr"].items())+"}}")
VIEWFIGS_JS=("const VIEWFIGS={survey:"+_figs_js(SURV["figs"])+",all:"+_figs_js(ALLV["figs"])+"};")
VIEWS_JS=("const VIEWS={survey:"+_json.dumps(SURV["data"])+",all:"+_json.dumps(ALLV["data"])+"};")
CONST_JS=("const GCOL="+_json.dumps(GCOL)+";const GORDER="+_json.dumps(GORDER)
        +";const SPCOL="+_json.dumps(SPECIES_COLOUR)+";const MAP_ALLCOL="+_json.dumps(MAP_ALLCOL)+";")
HANDLER_JS="""
const CFG={displayModeBar:false,responsive:true};
function isNarrow(){return window.innerWidth<=640;}
function cap1(s){return s.charAt(0).toUpperCase()+s.slice(1);}
function fmt(n){return Number(n).toLocaleString('en-GB');}
function colourFor(s){return SPCOL[s]||'#5c7065';}
function pad(n){return String(n).padStart(2,'0');}

/* ----- view state: official survey (Collections) vs all images ----- */
var VIEW='survey';
var SPECIES,STATIONS_BARS,CLOCK,CLOCK_ORDER,MAP_STATIONS,MAP_MAXTOT,MAP_SPMAX,SP_GUILD,MAP_BOUNDS,N_MAP,MAP_ORDER,MAPOPTS,CLOCKOPTS,KPI,SCALARS,FIG,SF;
function applyView(v){
  var d=VIEWS[v];
  SPECIES=d.SPECIES;STATIONS_BARS=d.STATIONS_BARS;CLOCK=d.CLOCK;CLOCK_ORDER=d.CLOCK_ORDER;
  MAP_STATIONS=d.MAP_STATIONS;MAP_MAXTOT=d.MAP_MAXTOT;MAP_SPMAX=d.MAP_SPMAX;SP_GUILD=d.SP_GUILD;
  MAP_BOUNDS=d.MAP_BOUNDS;N_MAP=d.N_MAP;MAP_ORDER=d.MAP_ORDER;MAPOPTS=d.mapOptions;CLOCKOPTS=d.clockOptions;
  KPI=d.KPI;SCALARS=d.scalars;FIG=VIEWFIGS[v];
  SF={T:{ov:FIG.phenoT_ov,dr:FIG.phenoT_dr},S:{ov:FIG.phenoS_ov,dr:FIG.phenoS_dr}};
  VIEW=v;recomputeGroups();
}
function renderKPIs(){var e=document.getElementById('kpis');if(!e||!KPI)return;
  e.innerHTML=KPI.map(function(k){return '<div class="kpi'+(k[2]?' alt':'')+'"><div class="v">'+k[0]+'</div><div class="k">'+k[1]+'</div></div>';}).join('');}
function fillSelect(id,opts,sel){var e=document.getElementById(id);if(!e||!opts)return;
  e.innerHTML=opts.map(function(o){var val=String(o[0]).replace(/&/g,'&amp;').replace(/"/g,'&quot;');return '<option value="'+val+'"'+(o[0]===sel?' selected':'')+'>'+o[1]+'</option>';}).join('');}
function renderScalars(){var s=SCALARS;if(!s)return;function S(id,v){var e=document.getElementById(id);if(e)e.innerHTML=v;}
  S('th_img',s.N_IMG+' photographs');S('th_d0',s.D0);S('th_d1',s.D1);S('th_det',s.N_DET+' animal and human detections');
  S('th_animal',s.N_ANIMAL);S('th_stn',s.N_STN);S('th_years',s.YEARS);
  S('ft_stn',s.N_STN);S('ft_img',s.N_IMG);S('ft_seq',s.N_SEQ);S('ft_active',s.ACTIVE);S('ft_d0',s.D0d);S('ft_d1',s.D1d);}
function rebuildForView(){
  SEASON.mode='T';SEASON.drill=null;
  var stog=document.getElementById('seasonTog');if(stog){var sb=stog.children;for(var i=0;i<sb.length;i++)sb[i].className=(sb[i].getAttribute('data-m')==='T'?'on':'');}
  var bk=document.getElementById('back_D');if(bk)bk.style.display='none';
  renderKPIs();renderScalars();
  fillSelect('mapSel',MAPOPTS,'All wildlife');fillSelect('clockSel',CLOCKOPTS,'All wildlife');
  renderSpecies();renderStations();
  renderGLegend('sbLegend',SB_GROUPS,sbHidden,'sb');renderGLegend('stLegend',ST_GROUPS,stHidden,'st');
  buildClock('All wildlife');renderSeason();buildMap('All wildlife');
}
function setScope(v){if(v===VIEW)return;sbHidden={};stHidden={};applyView(v);
  var tg=document.getElementById('scopeTog');if(tg){var bs=tg.children;for(var i=0;i<bs.length;i++)bs[i].className=(bs[i].getAttribute('data-v')===v?'on':'');}
  var nt=document.getElementById('scopeNote');if(nt)nt.innerHTML=(v==='survey'?'systematic survey only':'includes files not in Camera Trap Monitoring/Collections');
  rebuildForView();}

/* shared tooltip */
var tipEl=null;
function tipNode(){if(!tipEl){tipEl=document.createElement('div');tipEl.id='tip';document.body.appendChild(tipEl);}return tipEl;}
function showTip(html,e){var t=tipNode();t.innerHTML=html;t.style.opacity=1;var x=e.clientX+14,y=e.clientY+16;if(x+360>window.innerWidth)x=e.clientX-360;t.style.left=x+'px';t.style.top=y+'px';}
function hideTip(){if(tipEl)tipEl.style.opacity=0;}

/* ----- species bars: "what the cameras saw" (all taxa, no prioritisation) ----- */
var sbHidden={}, stHidden={};
var SB_GROUPS=[], ST_GROUPS=[];
function recomputeGroups(){
  SB_GROUPS=GORDER.filter(function(g){return SPECIES.some(function(d){return d.g===g;});});
  ST_GROUPS=GORDER.filter(function(g){return STATIONS_BARS.some(function(s){return s.segs[GORDER.indexOf(g)]>0;});});
}
function renderGLegend(hostId,groups,hidden,kind){
  var host=document.getElementById(hostId); if(!host)return;
  host.innerHTML=groups.map(function(g){return '<span class="gl" data-g="'+g+'" data-k="'+kind+'" style="opacity:'+(hidden[g]?0.32:1)+'"><i style="background:'+GCOL[g]+'"></i>'+cap1(g)+'</span>';}).join('');
  var sp=host.querySelectorAll('.gl');
  for(var i=0;i<sp.length;i++){(function(el){el.addEventListener('click',function(){toggleGroup(el.getAttribute('data-k'),el.getAttribute('data-g'));});})(sp[i]);}
}
function toggleGroup(kind,g){
  if(kind==='sb'){sbHidden[g]=!sbHidden[g];renderSpecies();renderGLegend('sbLegend',SB_GROUPS,sbHidden,'sb');}
  else{stHidden[g]=!stHidden[g];renderStations();renderGLegend('stLegend',ST_GROUPS,stHidden,'st');}
}
var sbMetric='events';
function renderSpecies(){
  var host=document.getElementById('speciesBars'); if(!host)return;
  var list=SPECIES.filter(function(d){return !sbHidden[d.g];}).sort(function(a,b){return b[sbMetric]-a[sbMetric];});
  var mx=Math.max.apply(null,list.map(function(d){return d[sbMetric];}).concat([1]));
  host.innerHTML=list.map(function(d){
    var col=GCOL[d.g]||'#5c7065', w=Math.max(1.4,d[sbMetric]/mx*100), nm=cap1(d.species);
    return '<div class="sbar" data-s="'+d.species+'"><div class="nm" title="'+nm+'">'+nm+'</div>'
      +'<div class="track"><div class="fill" style="width:'+w+'%;background:'+col+';opacity:'+(d.dom?0.5:1)+'"></div></div>'
      +'<div class="val">'+fmt(d[sbMetric])+'</div></div>';
  }).join('');
  var rows=host.querySelectorAll('.sbar');
  for(var i=0;i<rows.length;i++){(function(el){
    var d=SPECIES.filter(function(x){return x.species===el.getAttribute('data-s');})[0]; if(!d)return;
    var nm=cap1(d.species);
    el.addEventListener('mousemove',function(e){showTip('<div class="t">'+nm+' ('+cap1(d.g)+')'+(d.dom?' \\u00b7 domestic':'')+'</div>'
      +'<div class="rs"><span>Events <b>'+fmt(d.events)+'</b></span><span>Photos <b>'+fmt(d.images)+'</b></span>'
      +(d.mean_score!=null?'<span>Mean conf <b>'+d.mean_score+'</b></span>':'')+'</div>',e);});
    el.addEventListener('mouseleave',hideTip);
  })(rows[i]);}
}
(function(){var tg=document.getElementById('sbToggle'); if(tg)tg.addEventListener('click',function(e){
  var b=e.target.closest('button'); if(!b||!b.dataset.m)return; sbMetric=b.dataset.m;
  var ch=tg.children; for(var i=0;i<ch.length;i++)ch[i].classList.toggle('on',ch[i]===b); renderSpecies();});})();

/* ----- station bars: "busiest cameras" (all stations, stacked by group) ----- */
function renderStations(){
  var host=document.getElementById('stationBars'); if(!host)return;
  function vis(s){var t=0;for(var g=0;g<GORDER.length;g++){if(!stHidden[GORDER[g]])t+=s.segs[g];}return t;}
  var list=STATIONS_BARS.slice().sort(function(a,b){return vis(b)-vis(a);});
  var mx=Math.max.apply(null,list.map(vis).concat([1]));
  host.innerHTML=list.map(function(s){
    var segs='';
    for(var g=0;g<GORDER.length;g++){var v=s.segs[g]; if(v>0&&!stHidden[GORDER[g]])segs+='<div class="seg" style="width:'+(v/mx*100)+'%;background:'+GCOL[GORDER[g]]+'"></div>';}
    return '<div class="stbar" data-id="'+s.id+'"><div class="nm" title="'+s.id+'">'+s.id+'</div>'
      +'<div class="track">'+segs+'</div><div class="val">'+fmt(vis(s))+'</div></div>';
  }).join('');
  var rows=host.querySelectorAll('.stbar');
  for(var i=0;i<rows.length;i++){(function(el){
    var s=STATIONS_BARS.filter(function(x){return x.id===el.getAttribute('data-id');})[0]; if(!s)return;
    var parts=[],tot=0; for(var g=0;g<GORDER.length;g++){if(s.segs[g]>0&&!stHidden[GORDER[g]]){parts.push('<span style="color:'+GCOL[GORDER[g]]+'">'+cap1(GORDER[g])+' <b>'+fmt(s.segs[g])+'</b></span>');tot+=s.segs[g];}}
    var sub=(s.setup?'<div class="sub">'+s.setup+'</div>':'')+(s.lat!=null?'<div class="sub">'+s.lat+'\\u00b0N, '+s.lon+'\\u00b0E</div>':'');
    el.addEventListener('mousemove',function(e){showTip('<div class="t">'+s.id+' \\u00b7 '+fmt(tot)+' events</div>'+sub+'<div class="rs">'+parts.join('')+'</div>',e);});
    el.addEventListener('mouseleave',hideTip);
  })(rows[i]);}
}

/* ----- daily-rhythm ring (reference styling) ----- */
var curClock='All wildlife';
function buildClock(optKey){
  var o=CLOCK[optKey]; if(!o)return; curClock=optKey;
  var d=o.hr, col=(optKey==='All wildlife')?'#f0a839':colourFor(optKey), C=200,R0=78,RMAX=168;
  var mx=Math.max.apply(null,d.concat([1])), total=d.reduce(function(a,b){return a+b;},0);
  function P(r,deg){var a=(deg-90)*Math.PI/180; return [(C+r*Math.cos(a)).toFixed(1),(C+r*Math.sin(a)).toFixed(1)];}
  var s='<defs><filter id="ringglow"><feGaussianBlur stdDeviation="2.4"/></filter></defs>';
  var n1=P(RMAX+14,270), n2=P(RMAX+14,90);
  s+='<path d="M'+n1[0]+' '+n1[1]+' A '+(RMAX+14)+' '+(RMAX+14)+' 0 0 1 '+n2[0]+' '+n2[1]+' L '+C+' '+C+' Z" fill="#16233a" opacity="0.5"/>';
  [R0,(R0+RMAX)/2,RMAX].forEach(function(r){s+='<circle cx="'+C+'" cy="'+C+'" r="'+r+'" fill="none" stroke="#243029" stroke-width="1"/>';});
  for(var h=0;h<24;h+=3){var t=P(RMAX+26,h/24*360);
    s+='<text x="'+t[0]+'" y="'+(parseFloat(t[1])+4)+'" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="#647269">'+pad(h)+'</text>';}
  for(var k=0;k<24;k++){var len=R0+(d[k]/mx)*(RMAX-R0), deg=k/24*360, a=P(R0,deg), b=P(len,deg);
    s+='<line x1="'+a[0]+'" y1="'+a[1]+'" x2="'+b[0]+'" y2="'+b[1]+'" stroke="'+col+'" stroke-width="8" stroke-linecap="round" opacity="0.5" filter="url(#ringglow)"/>';
    s+='<line class="spoke" data-h="'+k+'" data-v="'+d[k]+'" x1="'+a[0]+'" y1="'+a[1]+'" x2="'+b[0]+'" y2="'+b[1]+'" stroke="'+col+'" stroke-width="7" stroke-linecap="round"/>';}
  s+='<text x="'+C+'" y="'+(C-6)+'" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="30" fill="#e8efe9">'+fmt(total)+'</text>';
  s+='<text x="'+C+'" y="'+(C+16)+'" text-anchor="middle" font-family="Inter, sans-serif" font-size="11" fill="#8fa39a">events</text>';
  var host=document.getElementById('fig_A');
  host.innerHTML='<div class="ring-legend"><span><i style="background:'+col+'"></i>Activity</span><span><i style="background:#2a3a55"></i>Night 18:00\\u201306:00</span></div>'
    +'<svg viewBox="0 0 400 400" class="dielring" role="img" aria-label="24-hour activity ring">'+s+'</svg>';
  var sp=host.querySelectorAll('.spoke');
  for(var i=0;i<sp.length;i++){(function(el){
    el.addEventListener('mousemove',function(e){showTip('<div class="t">'+pad(el.getAttribute('data-h'))+':00 \\u2013 '+pad((+el.getAttribute('data-h')+1)%24)+':00 \\u00b7 '+fmt(+el.getAttribute('data-v'))+' events</div>',e);el.setAttribute('stroke-width','10');});
    el.addEventListener('mouseleave',function(){hideTip();el.setAttribute('stroke-width','7');});
  })(sp[i]);}
  var cn=document.getElementById('clockN'); if(cn) cn.textContent='\\u00b7 '+fmt(total)+' timed detections';
}

/* ----- season: Timeline vs By-season, with group->species drilldown ----- */
var SEASON={mode:'T',drill:null};
/* SF is set by applyView(view) */
function seasonSpec(){var m=SF[SEASON.mode];return (SEASON.drill&&m.dr[SEASON.drill])?m.dr[SEASON.drill]:m.ov;}
function seasonClick(e){
  if(!e.points||!e.points.length)return;
  var pts=e.points.slice().sort(function(a,b){return a.curveNumber-b.curveNumber;});
  var ya=pts[0].yaxis, yData=NaN;
  try{
    var gd=document.getElementById('fig_D'), rect=gd.getBoundingClientRect();
    var py=((e.event&&e.event.clientY!=null)?e.event.clientY:rect.top)-rect.top-ya._offset;
    yData=ya.range[1]-(py/ya._length)*(ya.range[1]-ya.range[0]);
  }catch(err){}
  var total=0,i; for(i=0;i<pts.length;i++) total+=(pts[i].y||0);
  if(!isFinite(yData)||yData<0||yData>total) return;
  var cum=0,g=null;
  for(i=0;i<pts.length;i++){cum+=(pts[i].y||0); if(yData<=cum){var d=pts[i].data; g=d&&d.meta&&d.meta.guild; break;}}
  if(!g||!SF[SEASON.mode].dr[g])return;
  SEASON.drill=g;renderSeason();
}
function seasonLegendClick(e){
  if(SEASON.drill)return true;
  var tr=e.data&&e.data[e.curveNumber], g=tr&&tr.meta&&tr.meta.guild;
  if(g&&SF[SEASON.mode].dr[g]){SEASON.drill=g;renderSeason();}
  return false;
}
function renderSeason(){var sp=seasonSpec();var lay=JSON.parse(JSON.stringify(sp.layout));
  if(isNarrow()){lay.height=300;lay.margin={l:42,r:12,t:60,b:34};lay.font=lay.font||{};lay.font.size=11;
    if(lay.legend)lay.legend.font={size:10};if(lay.yaxis)lay.yaxis.title={text:''};}
  Plotly.purge('fig_D');
  Plotly.newPlot('fig_D',sp.data,lay,CFG).then(function(gd){if(!SEASON.drill){gd.on('plotly_click',seasonClick);gd.on('plotly_legendclick',seasonLegendClick);}});
  var b=document.getElementById('back_D');if(b)b.style.display=SEASON.drill?'inline-flex':'none';}
function setSeasonMode(m){SEASON.mode=m;SEASON.drill=null;
  var tg=document.getElementById('seasonTog'); if(tg){var ch=tg.children;for(var i=0;i<ch.length;i++)ch[i].classList.toggle('on',ch[i].dataset.m===m);}
  renderSeason();}
function back(panel){if(panel==='D'){SEASON.drill=null;renderSeason();}}

/* ----- camera-grid map: OpenStreetMap (default), Satellite, Terrain ----- */
var MAPOBJ=null, MAPMARKERS=[], curMap='All wildlife';
function popupHtml(s){
  var chips=s.top.length? s.top.map(function(t){return '<span class="chip" style="color:'+colourFor(t[0])+'">'+cap1(t[0])+' '+t[1]+'</span>';}).join('') : '';
  var rows='<div class="row"><span>Wildlife events</span><b>'+fmt(s.tot)+'</b></div>'
    +'<div class="row"><span>Species</span><b>'+s.spn+'</b></div>'
    +'<div class="row"><span>Sequences</span><b>'+fmt(s.seqs)+'</b></div>'
    +(s.score!=null?'<div class="row"><span>Mean conf.</span><b>'+s.score+'</b></div>':'')
    +(s.setup?'<div class="row"><span>Start</span><b>'+s.setup+'</b></div>':'')
    +(s.end?'<div class="row"><span>End</span><b>'+s.end+'</b></div>':'');
  return '<div class="pid">'+s.id+'</div><div class="stats">'+rows+'</div>'
    +(chips?'<div class="pri"><span class="plbl">Top 5</span>'+chips+'</div>':'<div class="sps">No animal detections</div>');
}
function drawMarkers(){
  if(!MAPOBJ)return;
  MAPMARKERS.forEach(function(m){MAPOBJ.removeLayer(m);}); MAPMARKERS=[];
  var isAll=(curMap==='All wildlife'), isAllH=(curMap==='All wildlife, humans and vehicles'), isAgg=isAll||isAllH;
  function valFor(s){return isAgg?(isAllH?s.toth:s.tot):(s.s[curMap]||0);}
  var mxAgg=Math.max.apply(null,MAP_STATIONS.map(valFor).concat([1]));
  var mxFlag=isAgg?1:(MAP_SPMAX[curMap]||1);
  var spc=isAgg?MAP_ALLCOL:colourFor(curMap);
  var ordered=MAP_STATIONS.slice().sort(function(a,b){return valFor(a)-valFor(b);});
  ordered.forEach(function(s){
    var val=valFor(s), active=isAgg?true:val>0;
    var mk=L.circleMarker([s.lat,s.lon],{
      radius:isAgg?(val<=0?4:4+Math.sqrt(val/mxAgg)*26):(active?5+Math.sqrt(val/mxFlag)*22:3.5),
      fillColor:active?spc:'#46554d', color:active?'#0c1310':'#2b352f',
      weight:active?1:0.6, fillOpacity:active?0.82:0.28});
    mk.bindPopup(popupHtml(s),{minWidth:320,maxWidth:360});
    mk.addTo(MAPOBJ); MAPMARKERS.push(mk);
  });
  var note=document.getElementById('mapNote');
  if(note){ if(isAgg){ note.innerHTML=N_MAP+' of '+N_MAP+' cameras located \\u00b7 circles sized by '+(isAllH?'wildlife, human and vehicle':'total wildlife')+' detections \\u00b7 tap a camera for detail'; }
    else { var n=MAP_STATIONS.filter(function(s){return (s.s[curMap]||0)>0;}).length;
      note.innerHTML=cap1(curMap)+' recorded at <b style="color:#e8efe9">'+n+'</b> of '+N_MAP+' cameras \\u00b7 circles sized by '+curMap+' detections'; } }
}
function buildMap(optKey){ curMap=optKey; drawMarkers(); }
function initMap(){
  var host=document.getElementById('fig_M');
  if(!host||typeof L==='undefined'||typeof MAP_STATIONS==='undefined'||!MAP_STATIONS.length||typeof MAP_BOUNDS==='undefined')return;
  MAPOBJ=L.map('fig_M',{scrollWheelZoom:false,zoomControl:true})
    .fitBounds([[MAP_BOUNDS.lat0,MAP_BOUNDS.lon0],[MAP_BOUNDS.lat1,MAP_BOUNDS.lon1]],{padding:[28,28]});
  var osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,subdomains:'abc',attribution:'\\u00a9 OpenStreetMap contributors'});
  var streets=L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',{maxZoom:20,subdomains:'abcd',attribution:'\\u00a9 OpenStreetMap contributors \\u00a9 CARTO'});
  var sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:17,attribution:'Tiles \\u00a9 Esri'});
  var topo=L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{maxZoom:17,attribution:'\\u00a9 OpenTopoMap (CC-BY-SA)'});
  topo.addTo(MAPOBJ);
  L.control.layers({'OpenStreetMap':osm,'Streets (offline-safe)':streets,'Satellite':sat,'Terrain':topo},null,{position:'topright'}).addTo(MAPOBJ);
  setTimeout(function(){MAPOBJ.invalidateSize();},60);
  buildMap('All wildlife');
}

/* ----- init + responsive ----- */
applyView('survey');
renderKPIs();
renderScalars();
/* selectors are server-rendered with the default (survey) options; JS refills only on scope switch */
renderSpecies();
renderStations();
renderGLegend('sbLegend',SB_GROUPS,sbHidden,'sb');
renderGLegend('stLegend',ST_GROUPS,stHidden,'st');
buildClock('All wildlife');
renderSeason();
(function(){var sn=document.getElementById('scopeNote');if(sn)sn.innerHTML='systematic survey only';})();
if(document.getElementById('fig_M')){initMap();}
var _rt,_lastN=isNarrow();
function _mapSize(){if(MAPOBJ){try{MAPOBJ.invalidateSize();}catch(e){}}}
function _resizeAll(){var el=document.getElementById('fig_D');if(el&&el.data){try{Plotly.Plots.resize(el);}catch(e){}}_mapSize();}
function _rebuildAll(){buildClock(curClock);renderSeason();renderSpecies();renderStations();buildMap(curMap);_mapSize();}
window.addEventListener('resize',function(){clearTimeout(_rt);_rt=setTimeout(function(){
  var n=isNarrow();if(n!==_lastN){_lastN=n;_rebuildAll();}else{_resizeAll();}
},220);});
window.addEventListener('orientationchange',function(){setTimeout(_rebuildAll,300);});
"""
PLOTLY_JS=get_plotlyjs()
# _LJS is set in the asset-loading block above
html=head+body+footer+(_LJS+"<script>"+PLOTLY_JS+"</script>\n<script>"+VIEWS_JS+VIEWFIGS_JS+CONST_JS+HANDLER_JS+"</script></body></html>")
with open(OUT,"w") as f: f.write(html)
print("written:",OUT,f"({os.path.getsize(OUT)/1024/1024:.2f} MB)  survey clock opts={len(SURV['data']['CLOCK_ORDER'])} all clock opts={len(ALLV['data']['CLOCK_ORDER'])}  drill[survey]={list(SURV['figs']['phenoT_dr'])}")
