from package_paths import SOURCE_ROOT, RUNS_ROOT, read_tile_scores
"""Replot the five main figures from frozen evidence, preserving originals."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,PercentFormatter
from PIL import Image

from package_paths import OUTPUT_ROOT as ROOT
SOURCE=SOURCE_ROOT
R5=SOURCE/'r5_endpoint_audit';R6=SOURCE/'r6_multiclass_audit'
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
QA=ROOT/'qa/figure_polish';QA.mkdir(parents=True,exist_ok=True)
DATA=OUT/'plotted_data';DATA.mkdir(exist_ok=True)
BLUE='#0072B2';RED='#C85B2D';TEAL='#138477';GOLD='#A67A16';PURPLE='#98639B'
INK='#233340';MUTED='#606F79';GRID='#E2E8EC';PALE='#EDF4F7';GRAY='#A7B2B9'
COL={'P1':BLUE,'P2':RED,'E1':PURPLE,'E2':TEAL,'E3':GOLD}
MARK={'P1':'o','P2':'s'}
REVIEW=ROOT/'qa/figure_layout';REVIEW.mkdir(parents=True,exist_ok=True)
MANUSCRIPT_WIDTH=6.5
MIN_TEXT_POINTS=9.0
PANEL_LABEL_OFFSET=(-24,16)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9.5,'axes.titlesize':10.5,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'axes.linewidth':.75,'axes.edgecolor':INK,'text.color':INK,'axes.labelcolor':INK,
    'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','figure.facecolor':'white','mathtext.default':'regular'})
manifest={'purpose':'Presentation revision; frozen data retained; no new fitted models',
          'figures':{},'sources':{},'software':{'matplotlib':mpl.__version__,'numpy':np.__version__,'pandas':pd.__version__}}

def read(path):
    path=Path(path);raw=path.read_bytes()
    manifest['sources'][str(path)]={'sha256':hashlib.sha256(raw).hexdigest()}
    return pd.read_csv(path,float_precision='round_trip')

def clean(ax,grid='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis=grid,color=GRID,lw=.65);ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.7,pad=4)

def title(ax,letter,text):
    ax.set_title(text,loc='left',pad=PANEL_LABEL_OFFSET[1],fontweight='bold')
    label=ax.annotate(letter,xy=(0,1),xycoords='axes fraction',
        xytext=PANEL_LABEL_OFFSET,textcoords='offset points',
        fontsize=12,fontweight='bold',va='baseline',annotation_clip=False)
    label.set_gid('panel-label')

def model_legend(fig,y=.963):
    handles=[Line2D([],[],color=COL[b],marker=MARK[b],ls='',ms=5.5,label=l) for b,l in [('P1','P1  U-Net'),('P2','P2  DOFA–UPerNet')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,y),ncol=2,frameon=False,columnspacing=2.2,handletextpad=.5)

def save(fig,n,notes):
    stem=f'Figure_{n}_MR'
    # Nine source points become 8.125 pt when a 7.2-in figure is placed at 6.5 in.
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text() and text.get_fontsize()<MIN_TEXT_POINTS:
            text.set_fontsize(MIN_TEXT_POINTS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fig.canvas.draw()
    glyph_warnings=[str(w.message) for w in caught if 'Glyph' in str(w.message) and 'missing' in str(w.message)]
    if glyph_warnings:raise RuntimeError(glyph_warnings)
    renderer=fig.canvas.get_renderer();bounds=fig.bbox
    outside=[]
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text():continue
        bb=text.get_window_extent(renderer)
        if bb.x0<bounds.x0-1 or bb.y0<bounds.y0-1 or bb.x1>bounds.x1+1 or bb.y1>bounds.y1+1:
            outside.append(text.get_text())
    if outside:raise RuntimeError(('Text outside figure',n,outside))
    tick_overlaps=[]
    for i,ax in enumerate(fig.axes):
        if not ax.axison:continue
        for name,ticks in [('x',ax.get_xticklabels()),('y',ax.get_yticklabels())]:
            visible=[t for t in ticks if t.get_visible() and t.get_text()]
            for j,a in enumerate(visible):
                ba=a.get_window_extent(renderer)
                for b in visible[j+1:]:
                    bb=b.get_window_extent(renderer)
                    if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>1 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>1:
                        tick_overlaps.append([i,name,a.get_text(),b.get_text()])
    if tick_overlaps:raise RuntimeError(('Overlapping tick labels',n,tick_overlaps))
    fig.savefig(OUT/(stem+'.pdf'))
    fig.savefig(OUT/(stem+'.svg'))
    fig.savefig(OUT/(stem+'.png'),dpi=250)
    fig.savefig(OUT/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    with Image.open(OUT/(stem+'.tiff')) as im:
        im.convert('RGB').save(OUT/(stem+'.tiff'),compression='tiff_lzw',dpi=(600,600))
    with Image.open(OUT/(stem+'.png')) as im:
        im.thumbnail((1800,1800));im.save(QA/(stem+'_preview.png'))
    visible_text=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    font_min=min(t.get_fontsize() for t in visible_text)
    report={'manuscript_width_inches':MANUSCRIPT_WIDTH,
        'manuscript_height_inches':MANUSCRIPT_WIDTH*fig.get_figheight()/fig.get_figwidth(),
        'source_font_min_pt':font_min,'manuscript_font_min_pt':font_min*MANUSCRIPT_WIDTH/fig.get_figwidth(),
        'text_outside_canvas':outside,'tick_label_overlaps':tick_overlaps,'missing_glyph_warnings':glyph_warnings,
        'panel_label_offset_points':list(PANEL_LABEL_OFFSET) if n!=1 else 'Custom schematic grid',
        'grayscale_review':'Pending visual inspection of PDF-derived render',
        'figure_sources':'Vector PDF/SVG and RGB TIFF; PNG is a preview'}
    (REVIEW/(stem+'_layout.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest['figures'][stem]={'inches':fig.get_size_inches().tolist(),'raster_dpi':600,'pdf_vector':True,'notes':notes,**report}
    plt.close(fig)

def figure1():
    fig=plt.figure(figsize=(7.2,5.8));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,720),ylim=(0,580));ax.axis('off')
    def txt(x,y,s,**kw):return ax.text(x,y,s,va='top',**kw)
    def heading(x,y,letter,text):
        txt(x-23,y,letter,fontsize=12,fontweight='bold');txt(x,y,text,fontsize=11,fontweight='bold')
    heading(50,558,'a','Fix the retained spatial selection')
    txt(50,530,'Lower risk',fontsize=9,color=MUTED)
    ax.add_patch(FancyArrowPatch((119,523),(259,523),arrowstyle='->',mutation_scale=9,lw=.85,color=MUTED))
    txt(269,530,'Higher risk',fontsize=9,color=MUTED)
    for x,col,label in [(403,BLUE,'Hazard'),(485,'#F9FBFC','Background'),(606,'#CCD5DA','Invalid')]:
        ax.add_patch(Rectangle((x,519),9,9,fc=col,ec=MUTED if label=='Background' else col,lw=.5))
        txt(x+14,531,label,fontsize=9,color=MUTED)
    ax.add_patch(Rectangle((45,395),473,106,fc='#F1F6F8',ec='none',zorder=0))
    # This schematic uses explicit pixel masks, not invented experimental observations.
    valid_rows=[8,6,8,7,8];hazard_counts=[4,12,8,12,36]
    masks=[];left=61;step=125;side=72;pixel=side/8;bottom=404
    for i,(vr,hc) in enumerate(zip(valid_rows,hazard_counts)):
        valid=np.zeros((8,8),bool);valid[:vr,:]=True
        coords=[(y,x) for y in range(vr) for x in range(8)]
        # A deterministic compact patch near a different edge in each tile.
        center=[(2,3),(3,5),(5,2),(3,3),(3,3)][i]
        coords.sort(key=lambda q:((q[0]-center[0])**2+(q[1]-center[1])**2,q))
        hazard=np.zeros((8,8),bool)
        for y,x in coords[:hc]:hazard[y,x]=True
        masks.append((valid,hazard))
        x0=left+i*step
        for y in range(8):
            for x in range(8):
                color=BLUE if hazard[y,x] else ('#F9FBFC' if valid[y,x] else '#CCD5DA')
                ax.add_patch(Rectangle((x0+x*pixel,bottom+(7-y)*pixel),pixel,pixel,fc=color,ec='white',lw=.18))
        ax.add_patch(Rectangle((x0,bottom),side,side,fc='none',ec=BLUE if i<4 else MUTED,lw=1.1,ls='-' if i<4 else '--'))
        txt(x0+side/2,496,f'Rank {i+1}',ha='center',fontsize=9.5,fontweight='bold' if i<4 else 'normal')
        txt(x0+side/2,395,f'{int(valid.sum())} valid\n{hc} hazard',ha='center',fontsize=9,color=MUTED,linespacing=1.0)
    ax.plot([left,left+3*step+side],[361,361],color=BLUE,lw=1.3)
    ax.plot([left,left],[361,367],color=BLUE,lw=1.3);ax.plot([left+3*step+side]*2,[361,367],color=BLUE,lw=1.3)
    txt(left+(3*step+side)/2,353,'Retain these four tiles',ha='center',color=BLUE,fontsize=9.5,fontweight='bold')
    txt(left+4*step+side/2,353,'Defer to review',ha='center',fontsize=9.5,color=MUTED)
    ax.plot([28,687],[327,327],color=GRID,lw=.8)
    heading(50,306,'b','Evaluate the retained set')
    heading(400,306,'c','Measure what is retained')
    ax.plot([365,365],[93,307],color=GRID,lw=.8)
    txt(50,266,'Tile mean',fontsize=10,fontweight='bold')
    txt(50,244,'Equal tile weights',fontsize=9,color=MUTED)
    txt(215,269,r'$\frac{1}{|S|}\sum_{i\in S}\ell(C_i)$',fontsize=14)
    ax.plot([50,338],[220,220],color=GRID,lw=.6)
    txt(50,207,'Event pooling',fontsize=10,fontweight='bold')
    txt(50,185,'Pool pixels, then\ncompute loss',fontsize=9,color=MUTED,linespacing=1.2)
    txt(215,209,r'$\ell\!\left(\sum_{i\in S}C_i\right)$',fontsize=14)
    txt(50,137,'Class convention',fontsize=10,fontweight='bold')
    txt(50,115,'Present classes or a fixed class set',fontsize=9,color=MUTED)
    txt(50,91,'S: retained tiles   ·   Cᵢ: confusion matrix   ·   ℓ: loss',fontsize=9,color=MUTED)
    labels=['Tile count','Valid area','Hazard area']
    retained=[4,sum(m[0].sum() for m in masks[:4]),sum(m[1].sum() for m in masks[:4])]
    total=[5,sum(m[0].sum() for m in masks),sum(m[1].sum() for m in masks)]
    colors=[INK,TEAL,BLUE]
    for i,(label,a,b,col) in enumerate(zip(labels,retained,total,colors)):
        y=267-i*63
        txt(400,y,label,fontsize=10,fontweight='bold')
        txt(677,y+2,f'{a/b:.1%}',ha='right',fontsize=12,fontweight='bold',color=col)
        ax.add_patch(Rectangle((400,y-29),277,12,color='#E8EDF0',lw=0))
        ax.add_patch(Rectangle((400,y-29),277*a/b,12,color=col,lw=0))
        unit='tiles' if i==0 else ('valid pixels' if i==1 else 'hazard pixels')
        txt(400,y-35,f'{a} / {b} {unit}',fontsize=9,color=MUTED)
    ax.plot([28,687],[65,65],color=GRID,lw=.8)
    txt(50,51,'Report the unit, loss, class set, coverage measure and budget.',fontsize=10,fontweight='bold')
    txt(50,27,'Schematic only · equal-area pixels · fixed tile scale and zoning',fontsize=9,color=MUTED)
    counts=pd.DataFrame({'rank':range(1,6),'valid_pixels':[int(m[0].sum()) for m in masks],'hazard_pixels':[int(m[1].sum()) for m in masks],'retained':[True]*4+[False]})
    counts.to_csv(DATA/'figure_1_illustrative_geometry.csv',index=False)
    save(fig,1,'Refined concept layout pairs evaluation choices with coverage denominators beneath the fixed selection. Illustrative masks, counts and coverage values unchanged; not observed study data. Development-history boundary remains in methods and caption.')

def scatter_deltas(ax,data,endpoints,count):
    for j,e in enumerate(endpoints):
        for block,offset in [('P1',-.15),('P2',.15)]:
            vals=data[(data.block==block)&(data.endpoint==e)].delta.to_numpy();assert len(vals)==count
            jitter=np.linspace(-.062,.062,len(vals))
            ax.scatter(j+offset+jitter,vals,s=14,marker=MARK[block],color=COL[block],alpha=.62,lw=0,zorder=2)
            ax.errorbar(j+offset,vals.mean(),yerr=vals.std(ddof=1),fmt='none',ecolor=COL[block],lw=1.0,capsize=2.5,zorder=3)
            ax.scatter(j+offset,vals.mean(),s=38,marker='D',color=COL[block],edgecolor='white',lw=.7,zorder=4)
    ax.axhline(0,color=INK,lw=.9);clean(ax)
    ax.set_xticks(range(len(endpoints)),[{'tile_mean_present_class_miou_risk':'Tile\nmIoU','present_class_miou_risk':'Pooled\nmIoU','foreground_iou_risk':'FG\nIoU','foreground_dice_risk':'FG\nDice'}[e] for e in endpoints])
    ax.set_xlim(-.48,len(endpoints)-.52)

def figure2():
    lo=read(R5/'calibration_loeo_endpoint_results.csv').pivot(index=['block','partition','seed','held_event','endpoint'],columns='method',values='aurc').reset_index()
    lo['delta']=lo.nonnegative_ridge-lo.soft_dice
    test=read(R5/'endpoint_aurc_by_run.csv');test=test[test.subset=='all_tiles'].pivot(index=['block','partition','seed','endpoint'],columns='score',values='event_macro_aurc').reset_index()
    test['delta']=test.georisk_v2-test.soft_dice_risk_binary
    curves=read(R5/'endpoint_risk_coverage_by_run_event.csv')
    curves=curves[(curves.subset=='all_tiles')&(curves.event_id=='__event_macro__')]
    c=curves.pivot(index=['block','partition','seed','endpoint','coverage'],columns='score',values='risk').reset_index();c['delta']=c.georisk_v2-c.soft_dice_risk_binary
    lo.to_csv(DATA/'figure_2_calibration_deltas.csv',index=False);test.to_csv(DATA/'figure_2_test_deltas.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.4));fig.subplots_adjust(left=.115,right=.975,bottom=.135,top=.85,wspace=.35,hspace=.59)
    model_legend(fig,.983)
    a,b=axes[0];title(a,'a','Calibration event audit');title(b,'b','Primary test runs')
    scatter_deltas(a,lo,['tile_mean_present_class_miou_risk','present_class_miou_risk','foreground_iou_risk','foreground_dice_risk'],33)
    scatter_deltas(b,test,['present_class_miou_risk','foreground_iou_risk','foreground_dice_risk'],9)
    a.set_ylabel('ΔAURC · ridge − self-overlap');b.set_ylabel('ΔAURC · v2 − self-overlap')
    a.set_ylim(-.24,.18);b.set_ylim(-.072,.029)
    a.text(.0,1.015,'33 correlated event diagnostics / model',transform=a.transAxes,fontsize=8.4,color=MUTED)
    b.text(.0,1.015,'9 correlated runs / model',transform=b.transAxes,fontsize=8.4,color=MUTED)
    panels=[]
    for ax,block,letter in zip(axes[1],['P1','P2'],['c','d']):
        title(ax,letter,block+' · effects across coverage')
        for endpoint,col,style,label in [('present_class_miou_risk',BLUE,'-','Pooled mIoU'),('foreground_dice_risk',TEAL,'--','Foreground Dice')]:
            g=c[(c.block==block)&(c.endpoint==endpoint)].groupby('coverage').delta.agg(mean='mean',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
            ax.fill_between(g.coverage,g.q25,g.q75,color=col,alpha=.09,lw=0)
            ax.plot(g.coverage,g['mean'],color=col,ls=style,lw=1.6,label=label,
                marker='o' if style=='-' else 's',markevery=(0 if style=='-' else 2,4),ms=3,mfc='white',mew=.8)
            g['block']=block;g['endpoint']=endpoint;panels.append(g)
        ax.axhline(0,color=INK,lw=.8);ax.axvline(.8,color=MUTED,ls=':',lw=1.0)
        ax.text(.79,.975,'80% target',transform=ax.get_xaxis_transform(),ha='right',va='top',fontsize=8.4,color=MUTED)
        ax.set(xlim=(.1,1),ylim=(-.14,.065),xlabel='Nominal tile-count coverage',ylabel='Δrisk · v2 − self-overlap')
        ax.set_xticks([.2,.4,.6,.8,1]);clean(ax)
    axes[1,0].legend(frameon=False,loc='lower left',fontsize=8.7)
    fig.text(.54,.027,'Diamonds/whiskers: mean ± SD   ·   Shading: run IQR   ·   Negative Δ favors v2/ridge',ha='center',fontsize=8.5,color=MUTED)
    pd.concat(panels).to_csv(DATA/'figure_2_curve_mean_iqr.csv',index=False)
    # Preserve the original curve means while aligning the plotted sign with Δ labels.
    old=read(R5/'coverage_gain_decomposition_endpoints.csv')
    for g in panels:
        q=old[(old.block==g.block.iloc[0])&(old.endpoint==g.endpoint.iloc[0])].sort_values('coverage')
        assert np.max(np.abs(q.delta_v2_minus_self_overlap.to_numpy()-g['mean'].to_numpy()))<1e-12
    save(fig,2,'Same original comparator data. Panels c/d use v2 minus self-overlap throughout; shaded IQR is descriptive across correlated runs, not a confidence interval.')

def figure3():
    cor=read(R5/'score_correlations_by_run_event.csv')
    ec=cor.groupby(['block','event_id','score','outcome']).spearman_rho.mean().reset_index()
    coupling=read(R5/'risk_content_coupling_summary.csv')
    run=read(R5/'endpoint_aurc_by_run.csv');run=run[(run.subset=='all_tiles')&(run.endpoint=='present_class_miou_risk')]
    fig=plt.figure(figsize=(7.2,6.1));gs=fig.add_gridspec(2,2,left=.12,right=.975,bottom=.12,top=.86,wspace=.38,hspace=.56,height_ratios=[1,1])
    model_legend(fig,.985);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    title(a,'a','Error and content alignment');title(b,'b','Event-level association');title(c,'c','Directional abundance baselines')
    combos=[('georisk_v2','tile_loss'),('georisk_v2','predicted_foreground_fraction'),('soft_dice_risk_binary','tile_loss'),('soft_dice_risk_binary','predicted_foreground_fraction')]
    for i,(score,outcome) in enumerate(combos):
        for block,offset in [('P1',-.15),('P2',.15)]:
            vals=ec[(ec.block==block)&(ec.score==score)&(ec.outcome==outcome)].spearman_rho.to_numpy();assert len(vals)==11
            a.scatter(i+offset+np.linspace(-.055,.055,11),vals,s=13,marker=MARK[block],color=COL[block],alpha=.58,lw=0)
            a.scatter(i+offset,vals.mean(),s=32,marker='D',color=COL[block],edgecolor='white',lw=.65,zorder=4)
    a.set(ylim=(-1.08,1.05),ylabel='Spearman ρ · event seed means',xlim=(-.45,3.45))
    a.set_xticks(range(4),['v2 ×\nloss','v2 ×\nFG','Self ×\nloss','Self ×\nFG']);a.axhline(0,color=INK,lw=.8);clean(a)
    a.text(0,1.015,'11 events / model; three seeds averaged',transform=a.transAxes,fontsize=8.4,color=MUTED)
    metrics=['foreground_coverage_delta_v2_minus_self_overlap_at_0_8','foreground_coverage_area_delta_v2_minus_self_overlap']
    for i,m in enumerate(metrics):
        for block,offset in [('P1',-.16),('P2',.16)]:
            r=coupling[(coupling.block==block)&(coupling.content_delta==m)].iloc[0]
            b.scatter(i+offset,r.spearman_rho,s=34,marker=MARK[block],color=COL[block])
            b.text(i+offset,r.spearman_rho+.037,f'{r.spearman_rho:.2f}',ha='center',fontsize=8.8)
    b.set(ylim=(0,.80),xlim=(-.48,1.48),ylabel='Spearman ρ with ΔAURC')
    b.set_xticks([0,1],['Hazard retention\nat 80% tiles','Integrated hazard\nretention']);clean(b)
    b.text(0,1.015,'11 events / model; exploratory',transform=b.transAxes,fontsize=8.4,color=MUTED)
    scores=['georisk_v2','soft_dice_risk_binary','predicted_foreground_fraction','predicted_foreground_scarcity']
    stats=[]
    for i,score in enumerate(scores):
        for block,offset in [('P1',-.17),('P2',.17)]:
            v=run[(run.block==block)&(run.score==score)].event_macro_aurc.to_numpy();assert len(v)==9
            c.scatter(i+offset+np.linspace(-.062,.062,len(v)),v,s=17,color=COL[block],marker=MARK[block],alpha=.60,lw=0,zorder=2)
            c.errorbar(i+offset,v.mean(),yerr=v.std(ddof=1),fmt='D',color=COL[block],mec='white',mew=.6,ms=5.2,lw=1.2,capsize=3,zorder=4)
            stats.append(dict(block=block,score=score,mean=v.mean(),sd=v.std(ddof=1),n=9))
    c.set(xlim=(-.48,3.48),ylim=(0,.54),ylabel='Event-macro AURC')
    c.set_xticks(range(4),['GeoRisk v2','Stored self-overlap','High-foreground\nrisk','Foreground\nscarcity']);clean(c)
    c.text(0,1.015,'Points: 9 correlated runs   ·   Diamonds and whiskers: mean ± SD',transform=c.transAxes,fontsize=8.5,color=MUTED)
    fig.text(.54,.025,'FG: predicted foreground fraction   ·   Self: stored self-overlap   ·   All analyses are post hoc',ha='center',fontsize=8.5,color=MUTED)
    ec.to_csv(DATA/'figure_3_event_seed_mean_correlations.csv',index=False);pd.DataFrame(stats).to_csv(DATA/'figure_3_aurc_mean_sd.csv',index=False)
    save(fig,3,'Panel a displays all 11 seed-averaged event correlations and their mean; panel b retains the four original correlations (uncorrected p values in caption); panel c replaces bars with individual runs, mean and SD.')

def figure4():
    raw=read(R5/'hazard_area_retention_by_run_event.csv');raw=raw[raw.score=='georisk_v2']
    ev=raw.groupby(['block','event_id']).agg(mean=('reference_foreground_coverage','mean'),minimum=('reference_foreground_coverage','min'),maximum=('reference_foreground_coverage','max'),count=('tile_count_coverage','mean')).reset_index()
    curves=read(R5/'hazard_coverage_curves.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.8));fig.subplots_adjust(left=.115,right=.975,bottom=.18,top=.90,wspace=.33,hspace=.44)
    order=['GHA','NGA','LKA','BOL','KHM','USA','PRY','ESP','SOM','IND','PAK']
    for ax,block,letter in zip(axes[0],['P1','P2'],['a','b']):
        title(ax,letter,block+' · event retention')
        g=ev[ev.block==block].set_index('event_id').loc[order];y=np.arange(len(order))[::-1]
        ax.errorbar(g['mean'],y,xerr=np.vstack([g['mean']-g.minimum,g.maximum-g['mean']]),fmt=MARK[block],color=COL[block],ms=4.2,capsize=2,lw=1.1,zorder=3)
        ax.scatter(g['count'],y,marker='|',color=INK,s=38,lw=1.3,zorder=4)
        ax.axvline(.8,color=MUTED,lw=.9,ls=':')
        ax.set(xlim=(.25,1.02),ylim=(-.6,10.6),xlabel='Reference-hazard coverage')
        ax.set_yticks(y,order);ax.set_xticks([.4,.6,.8,1]);clean(ax,'y')
    fig.text(.52,.975,'Nominal 80% tiles   ·   Dots: seed means   ·   Whiskers: seed range   ·   | : realized tile coverage',ha='center',fontsize=8.5,color=MUTED)
    styles=[('georisk_v2',BLUE,'-','o','GeoRisk v2'),('soft_dice_risk_binary',GOLD,(0,(5,2)),'s','Stored self-overlap'),('predicted_foreground_fraction',RED,(0,(2,2)),'v','High-foreground risk'),('predicted_foreground_scarcity',PURPLE,':','^','Foreground scarcity')]
    curveout=[]
    for ax,block,letter in zip(axes[1],['P1','P2'],['c','d']):
        title(ax,letter,block+' · count and hazard coverage')
        for si,(score,col,ls,marker,label) in enumerate(styles):
            g=curves[(curves.block==block)&(curves.score==score)].groupby('coverage').reference_foreground_coverage.mean().reset_index()
            ax.plot(g.coverage,g.reference_foreground_coverage,color=col,ls=ls,lw=1.7,label=label,
                marker=marker,markevery=(si%3,4),ms=3.5,mfc='white',mew=.8)
            g['block']=block;g['score']=score;curveout.append(g)
        ax.plot([.1,1],[.1,1],color=MUTED,ls='-.',lw=1,label='Count-proportional')
        ax.set(xlim=(.1,1),ylim=(0,1.025),xlabel='Nominal tile-count coverage',ylabel='Reference-hazard coverage')
        ax.set_xticks([.2,.4,.6,.8,1]);ax.set_yticks([0,.2,.4,.6,.8,1]);clean(ax)
    h,l=axes[1,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.52,.037),frameon=False,fontsize=8.7,columnspacing=1.4)
    ev.to_csv(DATA/'figure_4_event_mean_range.csv',index=False);pd.concat(curveout).to_csv(DATA/'figure_4_mean_curves.csv',index=False)
    save(fig,4,'Existing 11-event means and min–max over three seeds; curves average 33 correlated event-runs. Both model panels share the same axes and score styles.')

def figure5():
    metrics=read(SOURCE/'consolidated/georisk_v2_metrics.csv');decomp=read(R6/'e1_class_set_coverage_decomposition.csv')
    rows=[]
    for (block,partition,seed),g in metrics.groupby(['block','partition','seed']):
        p=g.set_index('score').event_macro_aurc;comp='self_dice_risk_multiclass' if block=='E1' else 'soft_dice_risk_binary'
        rows.append(dict(block=block,partition=partition,seed=seed,delta=p.georisk_v2-p[comp]))
    data=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.9));fig.subplots_adjust(left=.115,right=.975,bottom=.22,top=.82,wspace=.39)
    a,b=axes;title(a,'a','Effects across five blocks');title(b,'b','Kuro Siwo class-set sensitivity')
    for i,block in enumerate(['P1','P2','E1','E2','E3']):
        v=data[data.block==block].delta.to_numpy()
        # Reserve a separate horizontal lane for the mean so clustered E1 seeds remain visible.
        a.scatter(i+.1+np.linspace(-.09,.09,len(v)),v,s=25,color=COL[block],alpha=.72,edgecolor='white',lw=.4)
        a.scatter(i-.20,v.mean(),marker='D',s=36,color=INK,zorder=4)
    a.axhline(0,color=INK,lw=.9);a.axvline(1.5,color=GRAY,ls=':',lw=.8)
    a.set_xticks(range(5),['P1\nn=9','P2\nn=9','E1\nn=3','E2\nn=3','E3\nn=3'])
    a.set(xlim=(-.5,4.5),ylim=(-.116,.022),ylabel='ΔAURC · v2 − self-overlap');clean(a)
    a.text(.5,.016,'Primary',ha='center',fontsize=8.3,color=MUTED)
    a.text(3,.016,'Target-calibrated external',ha='center',fontsize=8.3,color=MUTED)
    b.plot(decomp.coverage,decomp.georisk_v2_class_set_shift,color=BLUE,lw=1.8,label='GeoRisk v2',marker='o',markevery=(0,4),ms=3.5,mfc='white',mew=.8)
    b.plot(decomp.coverage,decomp.self_overlap_class_set_shift,color=GOLD,lw=1.6,ls='--',label='Custom self-overlap',marker='s',markevery=(2,4),ms=3.5,mfc='white',mew=.8)
    q=decomp[decomp.coverage<=.5+1e-10]
    b.fill_between(q.coverage,0,q.mean_class_set_shift,color=BLUE,alpha=.07,lw=0)
    b.axvline(.8,ymax=.18,color=MUTED,ls=':',lw=.9)
    r=decomp.iloc[np.argmin(np.abs(decomp.coverage-.8))]
    b.scatter([r.coverage],[r.mean_class_set_shift],color=INK,s=22,zorder=4)
    b.annotate('80% tiles\nΔrisk = 0.0121',xy=(r.coverage,r.mean_class_set_shift),xytext=(.63,.13),fontsize=9,ha='left',arrowprops={'arrowstyle':'-','lw':.7,'color':MUTED})
    fraction=float(np.trapezoid(q.mean_class_set_shift,q.coverage)/np.trapezoid(decomp.mean_class_set_shift,decomp.coverage))
    b.text(.53,.34,f'{fraction:.1%} of integrated\nshift occurs by 50%',fontsize=9,ha='left',va='top')
    b.set(xlim=(.1,1),ylim=(0,.48),xlabel='Nominal tile-count coverage',ylabel='Fixed-class − present-class risk')
    b.set_xticks([.2,.4,.6,.8,1]);clean(b)
    b.legend(frameon=False,fontsize=8.5,loc='upper right',bbox_to_anchor=(1,1.02))
    fig.text(.54,.045,'Points: all correlated runs   ·   Diamonds: block means   ·   E1: custom multiclass comparator',ha='center',fontsize=8.3,color=MUTED)
    data.to_csv(DATA/'figure_5_block_paired_deltas.csv',index=False)
    save(fig,5,'All 27 block-run effects retained; three external seeds shown individually. Class-set shift curves unchanged; shaded area highlights the reported low-coverage region.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',type=int,choices=range(1,6));args=parser.parse_args()
    for n in ([args.only] if args.only else range(1,6)):globals()[f'figure{n}']()
    old=OUT/'figure_manifest.json'
    if args.only and old.exists():
        previous=json.loads(old.read_text());previous['figures'].update(manifest['figures']);previous['sources'].update(manifest['sources']);old.write_text(json.dumps(previous,indent=2),encoding='utf-8')
    else:old.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['figures'],indent=2))

if __name__=='__main__':main()
